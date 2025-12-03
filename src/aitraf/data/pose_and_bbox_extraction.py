"""Run YOLO pose + bounding box extraction on downloaded clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from ultralytics import YOLO

from aitraf.logging import logger


@dataclass
class PoseAndBBoxExtractionConfig:
    """Lightweight configuration for pose/bbox extraction."""

    clips_dir: Path | str
    poses_dir: Path | str
    boxes_dir: Path | str
    weights_path: Path | str
    device: str | int = 0
    imgsz: int = 640
    conf: float = 0.5
    force: bool = False
    limit: int | None = None

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
    clips = _list_clip_files(config.clips_dir)

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

        results = model(
            source=clip.as_posix(),
            device=config.device,
            imgsz=int(config.imgsz),
            conf=float(config.conf),
            verbose=False,
            stream=True,
        )
        try:
            pose_payload, box_payload = _pack_results(results)
            np.savez_compressed(pose_out, **pose_payload)
            np.savez_compressed(box_out, **box_payload)
        except Exception as exc:  # pragma: no cover - log only
            logger.warning("Failed to process {}: {}", clip, exc)
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
        "Pose/bbox extraction summary: {} processed, {} skipped (total {})",
        processed,
        skipped,
        len(clips),
    )


def _list_clip_files(clips_dir: Path) -> list[Path]:
    if not clips_dir.exists():
        return []
    clips: list[Path] = []
    for path in clips_dir.rglob("*"):
        if path.is_file():
            clips.append(path)
    return clips


def _pack_results(results: Iterable) -> tuple[dict, dict]:
    frame_indices: list[int] = []
    keypoints: list[np.ndarray] = []
    keypoint_scores: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    box_scores: list[np.ndarray] = []
    class_ids: list[np.ndarray] = []
    frame_shape: tuple[int, int] | None = None

    for frame_idx, result in enumerate(results):
        frame_indices.append(frame_idx)
        if frame_shape is None and getattr(result, "orig_shape", None):
            frame_shape = tuple(result.orig_shape)

        if getattr(result, "keypoints", None) is not None:
            keypoints.append(result.keypoints.xyn.cpu().numpy())
            keypoint_scores.append(result.keypoints.conf.cpu().numpy())
        else:
            keypoints.append(None)
            keypoint_scores.append(None)

        if getattr(result, "boxes", None) is not None:
            boxes.append(result.boxes.xyxy.cpu().numpy())
            box_scores.append(result.boxes.conf.cpu().numpy())
            class_ids.append(result.boxes.cls.cpu().numpy())
        else:
            boxes.append(None)
            box_scores.append(None)
            class_ids.append(None)

    pose_payload = {
        "frames": np.array(frame_indices, dtype=np.int32),
        "keypoints": np.array(keypoints, dtype=object),
        "keypoint_scores": np.array(keypoint_scores, dtype=object),
        "orig_shape": np.array(frame_shape or (), dtype=np.int32),
    }

    box_payload = {
        "frames": np.array(frame_indices, dtype=np.int32),
        "boxes_xyxy": np.array(boxes, dtype=object),
        "scores": np.array(box_scores, dtype=object),
        "class_ids": np.array(class_ids, dtype=object),
        "orig_shape": np.array(frame_shape or (), dtype=np.int32),
    }
    return pose_payload, box_payload


__all__ = ["PoseAndBBoxExtractionConfig", "pose_and_bbox_extraction"]
