"""Load and validate extracted pose artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

KEYPOINT_COUNT = 17
COORDINATE_COUNT = 2
REQUIRED_KEYS = ("frames", "keypoints", "scores")


class PoseArtifactError(RuntimeError):
    """Raised when a pose artifact is missing or does not match the schema."""


@dataclass(frozen=True)
class PoseArtifact:
    """Per-frame pose detections for one clip, normalized to the rotated frame."""

    frames: np.ndarray
    keypoints: np.ndarray
    scores: np.ndarray

    def __len__(self) -> int:
        return len(self.frames)


def load_pose_artifact(path: Path) -> PoseArtifact:
    """Read a pose `.npz` and reject anything that is not the expected shape.

    `allow_pickle` is required because the extraction pipeline stores ragged
    per-frame detections as object arrays. These artifacts are produced by this
    repository and read only from the private bucket.
    """
    if not path.is_file():
        raise PoseArtifactError(f"Pose artifact not found: {path}")

    try:
        payload = np.load(path, allow_pickle=True)
    except Exception as exc:
        raise PoseArtifactError(f"Unreadable pose artifact: {path}") from exc

    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise PoseArtifactError(
            f"Pose artifact {path} is missing required keys: {', '.join(missing)}"
        )

    frames = payload["frames"]
    keypoints = payload["keypoints"]
    scores = payload["scores"]

    _validate_frames(frames, path)
    _validate_lengths(frames, keypoints, scores, path)
    _validate_detections(keypoints, scores, path)

    return PoseArtifact(frames=frames, keypoints=keypoints, scores=scores)


def _validate_frames(frames: np.ndarray, path: Path) -> None:
    if frames.ndim != 1:
        raise PoseArtifactError(f"Pose artifact {path} has non-1-D frames")
    if frames.dtype != np.int32:
        raise PoseArtifactError(
            f"Pose artifact {path} has frames dtype {frames.dtype}, expected int32"
        )
    if not np.array_equal(frames, np.arange(len(frames), dtype=np.int32)):
        raise PoseArtifactError(
            f"Pose artifact {path} frames are not dense and 0-based"
        )


def _validate_lengths(
    frames: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, path: Path
) -> None:
    if len(keypoints) != len(frames) or len(scores) != len(frames):
        raise PoseArtifactError(
            f"Pose artifact {path} length mismatch: {len(frames)} frames, "
            f"{len(keypoints)} keypoint entries, {len(scores)} score entries"
        )


def _validate_detections(keypoints: np.ndarray, scores: np.ndarray, path: Path) -> None:
    for index, (frame_keypoints, frame_scores) in enumerate(zip(keypoints, scores)):
        frame_keypoints = np.asarray(frame_keypoints)
        frame_scores = np.asarray(frame_scores)

        if frame_keypoints.shape[-2:] != (KEYPOINT_COUNT, COORDINATE_COUNT):
            raise PoseArtifactError(
                f"Pose artifact {path} frame {index} has keypoint shape "
                f"{frame_keypoints.shape}, expected trailing "
                f"({KEYPOINT_COUNT}, {COORDINATE_COUNT})"
            )
        if frame_scores.shape[-1:] != (KEYPOINT_COUNT,):
            raise PoseArtifactError(
                f"Pose artifact {path} frame {index} has score shape "
                f"{frame_scores.shape}, expected trailing ({KEYPOINT_COUNT},)"
            )
        if frame_keypoints.shape[0] != frame_scores.shape[0]:
            raise PoseArtifactError(
                f"Pose artifact {path} frame {index} detection count mismatch: "
                f"{frame_keypoints.shape[0]} keypoint sets, "
                f"{frame_scores.shape[0]} score sets"
            )


__all__ = [
    "PoseArtifact",
    "PoseArtifactError",
    "load_pose_artifact",
]
