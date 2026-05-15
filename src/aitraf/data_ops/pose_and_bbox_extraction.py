"""Run YOLO pose + bounding box extraction on downloaded clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import imageio.v3 as iio
import numpy as np
from ultralytics import YOLO

from aitraf.logging import logger
from aitraf.data_ops.utils import list_clip_files
from aitraf.utils.video_utils import get_video_rotation_deg


@dataclass
class PoseAndBBoxExtractionConfig:
    """Lightweight configuration for pose/bbox extraction."""

    clips_dir: Path | str
    poses_dir: Path | str
    boxes_dir: Path | str
    weights_path: Path | str
    device: str
    imgsz: int
    conf: float
    batch_size: int
    max_det: int
    force: bool
    limit: int | None

    def __post_init__(self) -> None:
        self.clips_dir = Path(self.clips_dir)
        self.poses_dir = Path(self.poses_dir)
        self.boxes_dir = Path(self.boxes_dir)
        candidate = Path(self.weights_path)
        if candidate.exists():
            self.weights_path = candidate
        else:
            self.weights_path = str(self.weights_path)


def pose_and_bbox_extraction(config: PoseAndBBoxExtractionConfig) -> None:
    clips = list_clip_files(config.clips_dir)

    if config.limit is not None:
        clips = clips[: int(config.limit)]

    if not clips:
        logger.info("No clips found in {}", config.clips_dir)
        return

    config.poses_dir.mkdir(parents=True, exist_ok=True)
    config.boxes_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(config.weights_path))
    total = len(clips)
    logger.info("Running pose/bbox extraction on {} clips", total)

    processed = 0
    skipped = 0
    errors = 0
    progress_step = max(1, total // 10)

    for idx, clip in enumerate(clips, start=1):
        pose_out = config.poses_dir / f"{clip.stem}.npz"
        box_out = config.boxes_dir / f"{clip.stem}.npz"
        if not config.force and pose_out.exists() and box_out.exists():
            skipped += 1
            continue
        if not clip.exists():
            logger.warning("Missing clip {}, skipping", clip)
            continue

        try:
            rotation_deg = get_video_rotation_deg(clip)
            results = _predict_clip_batches(clip, rotation_deg, model, config)
            pose_payload, box_payload = _prepare_results(results)

            np.savez_compressed(pose_out, **pose_payload)
            np.savez_compressed(box_out, **box_payload)

        except Exception as exc:  # pragma: no cover - log only
            logger.warning("Failed to process {}: {}", clip, exc)
            errors += 1
            continue

        processed += 1

        if idx == len(clips) or idx % progress_step == 0:
            pct = (idx / len(clips)) * 100
            logger.info(
                "Pose/bbox extraction progress: {}/{} ({:.1f}%)",
                idx,
                len(clips),
                pct,
            )

    logger.info(
        "Pose/bbox extraction summary: {} processed, {} skipped, {} errors (total {})",
        processed,
        skipped,
        errors,
        len(clips),
    )


def _prepare_results(results: Iterable) -> tuple[dict, dict]:
    frame_indices: list[int] = []
    keypoints: list[np.ndarray] = []
    keypoint_scores: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    box_scores: list[np.ndarray] = []

    for frame_idx, result in enumerate(results):
        frame_indices.append(frame_idx)
        kpts = result.keypoints
        boxes_obj = result.boxes

        keypoints.append(kpts.xyn.cpu().numpy())
        keypoint_scores.append(kpts.conf.cpu().numpy())

        boxes.append(boxes_obj.xyxyn.cpu().numpy())
        box_scores.append(boxes_obj.conf.cpu().numpy())

    pose_payload = {
        "frames": np.array(frame_indices, dtype=np.int32),
        "keypoints": np.array(keypoints, dtype=object),
        "scores": np.array(keypoint_scores, dtype=object),
    }

    box_payload = {
        "frames": np.array(frame_indices, dtype=np.int32),
        "boxes": np.array(boxes, dtype=object),
        "scores": np.array(box_scores, dtype=object),
    }

    return pose_payload, box_payload


def _predict_clip_batches(
    clip_path: Path,
    rotation_deg: int,
    model: YOLO,
    config: PoseAndBBoxExtractionConfig,
):
    frames = _iter_frames(clip_path, rotation_deg)

    for batch in _batched(frames, int(config.batch_size)):
        yield from _predict_frames(batch, model, config)


def _iter_frames(clip_path: Path, rotation_deg: int) -> Iterable[np.ndarray]:
    rotate_quarter_turns = rotation_deg // 90

    for frame in iio.imiter(str(clip_path)):
        if rotate_quarter_turns:
            yield np.rot90(frame, k=rotate_quarter_turns)
        else:
            yield frame


def _batched(
    frames: Iterable[np.ndarray], batch_size: int
) -> Iterable[list[np.ndarray]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: list[np.ndarray] = []

    for frame in frames:
        batch.append(frame)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def _predict_frames(
    frames: list[np.ndarray], model: YOLO, config: PoseAndBBoxExtractionConfig
):
    return model.predict(
        source=frames,
        device=config.device,
        imgsz=int(config.imgsz),
        conf=float(config.conf),
        max_det=int(config.max_det),
        verbose=False,
    )


__all__ = ["PoseAndBBoxExtractionConfig", "pose_and_bbox_extraction"]
