"""Extract cached VideoMAE segment features for temporal AQA heads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor

from aitraf.logging import logger
from aitraf.data_ops.utils import list_clip_files
from aitraf.processing.models.video_mae_temporal_fusion import (
    video_feature_cache_path,
)
from aitraf.processing.video import load_segmented_video_frames

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

    def __post_init__(self) -> None:
        self.clips_dir = Path(self.clips_dir)
        self.features_dir = Path(self.features_dir)
        self.model_cache_dir = Path(self.model_cache_dir)
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
        "Extracting VideoMAE features for {} clips ({} skipped from cache)",
        len(pending_clips),
        skipped,
    )
    if not pending_clips:
        logger.info(
            "VideoMAE feature extraction summary: 0 processed, {} skipped, 0 errors (total {})",
            skipped,
            total,
        )
        return

    dataset = VideoMaeFeatureDataset(
        clips=pending_clips,
        processor=processor,
        config=config,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=str(config.device).startswith("cuda"),
        persistent_workers=config.num_workers > 0,
        collate_fn=_collate_feature_samples,
    )

    processed = 0
    errors = 0
    progress_step = max(1, total // 10)
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
            features = _extract_batch_features(
                pixel_values=batch["pixel_values"],
                model=model,
                device=config.device,
                num_clips=config.num_clips,
            )
            saved, save_errors = _save_batch_features(features, batch, config)
            processed += saved
            errors += save_errors

        completed = skipped + processed + errors
        if completed >= next_progress or completed == total:
            pct = (completed / total) * 100
            logger.info(
                "VideoMAE feature extraction progress: {}/{} ({:.1f}%)",
                completed,
                total,
                pct,
            )
            while next_progress <= completed:
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
        features, frame_indices = _extract_clip_features(
            video_id=video_id,
            processor=processor,
            model=model,
            config=config,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "video_id": video_id,
                "backbone": config.backbone,
                "num_clips": config.num_clips,
                "sample_frames": config.sample_frames,
                "sampling_dist": config.sampling_dist,
                "frame_indices": frame_indices,
                "features": features.half().cpu(),
            },
            output_path,
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
            pixel_values, frame_indices = _prepare_clip_pixel_values(
                video_id=video_id,
                processor=self.processor,
                config=self.config,
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


def _extract_clip_features(
    *,
    video_id: str,
    processor: VideoMAEImageProcessor,
    model: torch.nn.Module,
    config: VideoMaeFeatureExtractionConfig,
) -> tuple[torch.Tensor, list[list[int]]]:
    clip_pixel_values, frame_indices = _prepare_clip_pixel_values(
        video_id=video_id,
        processor=processor,
        config=config,
    )
    clip_pixel_values = clip_pixel_values.to(config.device)

    with torch.inference_mode():
        outputs = model(pixel_values=clip_pixel_values)
        features = outputs.last_hidden_state.mean(dim=1)
    return features, frame_indices


def _prepare_clip_pixel_values(
    *,
    video_id: str,
    processor: VideoMAEImageProcessor,
    config: VideoMaeFeatureExtractionConfig,
) -> tuple[torch.Tensor, list[list[int]]]:
    segments, frame_indices = load_segmented_video_frames(
        video_id=video_id,
        local_clips_dir=config.clips_dir,
        num_segments=config.num_clips,
        frames_per_segment=config.sample_frames,
        sampling_dist=config.sampling_dist,
    )
    clip_pixel_values = processor(segments, return_tensors="pt")["pixel_values"]
    return clip_pixel_values, frame_indices


def _extract_batch_features(
    *,
    pixel_values: torch.Tensor,
    model: torch.nn.Module,
    device: str,
    num_clips: int,
) -> torch.Tensor:
    batch_size = pixel_values.shape[0]
    flat_pixel_values = pixel_values.reshape(
        batch_size * num_clips,
        *pixel_values.shape[2:],
    ).to(device, non_blocking=True)
    with torch.inference_mode():
        outputs = model(pixel_values=flat_pixel_values)
        features = outputs.last_hidden_state.mean(dim=1)
    return features.reshape(batch_size, num_clips, -1).half().cpu()


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
        features_dir=config.features_dir,
        video_id=video_id,
        backbone=config.backbone,
        num_clips=config.num_clips,
        sample_frames=config.sample_frames,
        sampling_dist=config.sampling_dist,
    )


__all__ = [
    "VideoMaeFeatureExtractionConfig",
    "VideoMaeFeatureDataset",
    "extract_all_clip_features",
    "extract_clip_features",
    "video_mae_feature_extraction",
]
