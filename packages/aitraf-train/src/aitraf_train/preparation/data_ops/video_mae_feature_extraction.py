"""Extract cached VideoMAE segment features for temporal AQA heads."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from aitraf_core.loading import load_huggingface_model
from aitraf_core.pre_processing import (
    predict_segment_feature_vectors_with_cache,
    video_feature_cache_dir,
)
from aitraf_train.logging import logger
from aitraf_train.preparation.data_ops.utils import list_clip_files


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

    if not clips:
        logger.info("No clips found in {}", config.clips_dir)
        return

    feature_extractor = load_huggingface_model(
        model_name=config.backbone,
        model_cache_dir=config.model_cache_dir,
        device=config.device,
        config_kwargs={"num_frames": config.sample_frames},
    )

    total = len(clips)
    counts: Counter[str] = Counter()
    progress_step = max(1, total // 10)

    logger.info("Extracting VideoMAE features for {} clips", total)

    for idx, clip in enumerate(clips, start=1):
        video_id = str(clip.relative_to(config.clips_dir))
        feature_path = config.feature_cache_dir / Path(video_id).with_suffix(".pt")

        try:
            predict_segment_feature_vectors_with_cache(
                video_id=video_id,
                clips_dir=config.clips_dir,
                feature_path=feature_path,
                feature_extractor=feature_extractor,
                num_clips=config.num_clips,
                sample_frames=config.sample_frames,
                sampling_dist=config.sampling_dist,
                force=config.force,
                on_cache_hit=lambda _: counts.update(skipped=1),
                on_cache_write=lambda _: counts.update(processed=1),
            )
        except Exception as exc:  # pragma: no cover - log only
            logger.warning("Failed to extract VideoMAE features for {}: {}", clip, exc)
            counts.update(errors=1)
            continue

        completed = counts["processed"] + counts["skipped"] + counts["errors"]
        if idx == total or completed % progress_step == 0:
            pct = (completed / total) * 100
            logger.info(
                "VideoMAE feature extraction progress: {}/{} ({:.1f}%)",
                completed,
                total,
                pct,
            )

    logger.info(
        "VideoMAE feature extraction summary: {} processed, {} skipped, {} errors (total {})",
        counts["processed"],
        counts["skipped"],
        counts["errors"],
        total,
    )


__all__ = [
    "VideoMaeFeatureExtractionConfig",
    "video_mae_feature_extraction",
]
