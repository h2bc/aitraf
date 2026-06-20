"""Extract cached VideoMAE segment features for temporal AQA heads."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor

from aitraf_train.logging import logger
from aitraf_train.data_ops.utils import list_clip_files
from aitraf_core.pre_processing import (
    cached_video_mae_feature_extraction,
    video_feature_cache_dir,
    video_feature_cache_path,
)
from aitraf_core.pre_processing.models.video_mae import (
    VideoMaeFeatureExtractor,
    extract_video_mae_batch_features,
    prepare_clip_pixel_values,
)
ClipFeatureExtractionResult = Literal["processed", "skipped", "error"]


@dataclass
class VideoMaeFeatureExtractionConfig:
    clips_dir: Path | str
    features_dir: Path | str
    backbone: str
    model_cache_dir: Path | str
    device: str
    batch_size: int
    num_workers: int
    num_clips: int
    sample_frames: int
    sampling_dist: str
    force: bool
    limit: int | None
    feature_cache_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.clips_dir = Path(self.clips_dir)
        self.features_dir = Path(self.features_dir)
        self.model_cache_dir = Path(self.model_cache_dir)
        self.feature_cache_dir = video_feature_cache_dir(
            features_dir=self.features_dir,
            backbone=self.backbone,
            num_clips=self.num_clips,
            sample_frames=self.sample_frames,
            sampling_dist=self.sampling_dist,
        )
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


def video_mae_feature_extraction(config: VideoMaeFeatureExtractionConfig) -> None:
    clips = list_clip_files(config.clips_dir)
    if config.limit is not None:
        clips = clips[: int(config.limit)]

    processor = VideoMAEImageProcessor.from_pretrained(
        config.backbone,
        cache_dir=str(config.model_cache_dir),
    )
    backbone_config = AutoConfig.from_pretrained(
        config.backbone,
        cache_dir=str(config.model_cache_dir),
        trust_remote_code=True,
        num_frames=config.sample_frames,
    )
    model = AutoModel.from_pretrained(
        config.backbone,
        config=backbone_config,
        cache_dir=str(config.model_cache_dir),
        trust_remote_code=True,
    ).to(config.device)
    model.eval()

    extract_all_clip_features(
        clips=clips,
        processor=processor,
        model=model,
        config=config,
    )


def extract_all_clip_features(
    *,
    clips: list[Path],
    processor: VideoMAEImageProcessor,
    model: torch.nn.Module,
    config: VideoMaeFeatureExtractionConfig,
) -> None:
    if not clips:
        logger.info("No clips found in {}", config.clips_dir)
        return

    total = len(clips)
    pending_clips, skipped = _filter_pending_clips(clips, config)
    logger.info(
        "VideoMAE feature extraction cache scan: {} total clips, {} cached, {} pending",
        total,
        skipped,
        len(pending_clips),
    )
    if not pending_clips:
        logger.info(
            "VideoMAE feature extraction summary: 0 processed, {} skipped, 0 errors (total {})",
            skipped,
            total,
        )
        return

    logger.info(
        "Extracting VideoMAE features for {} pending clips",
        len(pending_clips),
    )

    dataset = VideoMaeFeatureDataset(
        clips=pending_clips,
        processor=processor,
        config=config,
    )
    dataloader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "shuffle": False,
        "pin_memory": str(config.device).startswith("cuda"),
        "collate_fn": _collate_feature_samples,
    }
    if config.num_workers > 0:
        dataloader_kwargs |= {
            "persistent_workers": False,
            "prefetch_factor": 1,
        }
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    processed = 0
    errors = 0
    pending_total = len(pending_clips)
    progress_step = max(1, pending_total // 10)
    next_progress = progress_step

    for batch in dataloader:
        for item in batch["errors"]:
            logger.warning(
                "Failed to prepare VideoMAE features for {}: {}",
                item["clip"],
                item["error"],
            )
        errors += len(batch["errors"])

        if "pixel_values" in batch:
            features = extract_video_mae_batch_features(
                pixel_values=batch["pixel_values"],
                model=model,
                device=config.device,
                num_clips=config.num_clips,
            )
            saved, save_errors = _save_batch_features(features, batch, config)
            processed += saved
            errors += save_errors

        completed_pending = processed + errors
        if completed_pending >= next_progress or completed_pending == pending_total:
            pending_pct = (completed_pending / pending_total) * 100
            total_completed = skipped + completed_pending
            total_pct = (total_completed / total) * 100
            logger.info(
                "VideoMAE feature extraction progress: {}/{} pending ({:.1f}%), {}/{} total ready ({:.1f}%)",
                completed_pending,
                pending_total,
                pending_pct,
                total_completed,
                total,
                total_pct,
            )
            while next_progress <= completed_pending:
                next_progress += progress_step

    logger.info(
        "VideoMAE feature extraction summary: {} processed, {} skipped, {} errors (total {})",
        processed,
        skipped,
        errors,
        total,
    )


def extract_clip_features(
    *,
    clip: Path,
    processor: VideoMAEImageProcessor,
    model: torch.nn.Module,
    config: VideoMaeFeatureExtractionConfig,
) -> ClipFeatureExtractionResult:
    video_id = str(clip.relative_to(config.clips_dir))
    output_path = _feature_output_path(video_id, config)
    if output_path.exists() and not config.force:
        return "skipped"

    try:
        cached_video_mae_feature_extraction(
            video_id=video_id,
            clips_dir=config.clips_dir,
            feature_path=output_path,
            feature_extractor=VideoMaeFeatureExtractor(
                processor=processor,
                model=model,
                device=config.device,
            ),
            backbone=config.backbone,
            num_clips=config.num_clips,
            sample_frames=config.sample_frames,
            sampling_dist=config.sampling_dist,
            cache_video_features=True,
            force=config.force,
        )
    except Exception as exc:  # pragma: no cover - log only
        logger.warning("Failed to extract VideoMAE features for {}: {}", clip, exc)
        return "error"

    return "processed"


class VideoMaeFeatureDataset(Dataset):
    def __init__(
        self,
        *,
        clips: list[Path],
        processor: VideoMAEImageProcessor,
        config: VideoMaeFeatureExtractionConfig,
    ) -> None:
        self.clips = clips
        self.processor = processor
        self.config = config

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict:
        clip = self.clips[index]
        video_id = str(clip.relative_to(self.config.clips_dir))
        output_path = _feature_output_path(video_id, self.config)

        try:
            pixel_values, frame_indices = prepare_clip_pixel_values(
                video_id=video_id,
                processor=self.processor,
                clips_dir=self.config.clips_dir,
                num_clips=self.config.num_clips,
                sample_frames=self.config.sample_frames,
                sampling_dist=self.config.sampling_dist,
            )
        except Exception as exc:  # pragma: no cover - worker-safe reporting
            return {
                "status": "error",
                "clip": str(clip),
                "error": str(exc),
            }

        return {
            "status": "ready",
            "clip": str(clip),
            "video_id": video_id,
            "output_path": str(output_path),
            "frame_indices": frame_indices,
            "pixel_values": pixel_values,
        }


def _save_batch_features(
    features: torch.Tensor,
    batch: dict,
    config: VideoMaeFeatureExtractionConfig,
) -> tuple[int, int]:
    saved = 0
    errors = 0
    for idx, video_features in enumerate(features):
        output_path = Path(batch["output_paths"][idx])
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "video_id": batch["video_ids"][idx],
                    "backbone": config.backbone,
                    "num_clips": config.num_clips,
                    "sample_frames": config.sample_frames,
                    "sampling_dist": config.sampling_dist,
                    "frame_indices": batch["frame_indices"][idx],
                    "features": video_features,
                },
                output_path,
            )
            saved += 1
        except Exception as exc:  # pragma: no cover - log only
            logger.warning(
                "Failed to save VideoMAE features for {}: {}",
                batch["clips"][idx],
                exc,
            )
            errors += 1
    return saved, errors


def _collate_feature_samples(samples: list[dict]) -> dict:
    ready = [sample for sample in samples if sample["status"] == "ready"]
    errors = [sample for sample in samples if sample["status"] == "error"]

    batch: dict = {"errors": errors}
    if ready:
        batch |= {
            "clips": [sample["clip"] for sample in ready],
            "video_ids": [sample["video_id"] for sample in ready],
            "output_paths": [sample["output_path"] for sample in ready],
            "frame_indices": [sample["frame_indices"] for sample in ready],
            "pixel_values": torch.stack(
                [sample["pixel_values"] for sample in ready],
                dim=0,
            ),
        }
    return batch


def _filter_pending_clips(
    clips: list[Path],
    config: VideoMaeFeatureExtractionConfig,
) -> tuple[list[Path], int]:
    if config.force:
        return clips, 0

    pending: list[Path] = []
    skipped = 0
    for clip in clips:
        video_id = str(clip.relative_to(config.clips_dir))
        output_path = _feature_output_path(video_id, config)
        if output_path.exists():
            skipped += 1
        else:
            pending.append(clip)
    return pending, skipped


def _feature_output_path(
    video_id: str,
    config: VideoMaeFeatureExtractionConfig,
) -> Path:
    return video_feature_cache_path(
        feature_cache_dir=config.feature_cache_dir,
        video_id=video_id,
    )


__all__ = [
    "VideoMaeFeatureExtractionConfig",
    "VideoMaeFeatureDataset",
    "extract_all_clip_features",
    "extract_clip_features",
    "video_mae_feature_extraction",
]
