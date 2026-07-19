from pathlib import Path

import numpy as np
import pytest

from aitraf_core.render.poses import (
    PoseArtifactError,
    load_pose_artifact,
)


def _write(path: Path, **payload) -> Path:
    np.savez_compressed(path, **payload)
    return path.with_suffix(".npz") if path.suffix != ".npz" else path


def _valid_payload(n_frames: int = 3, n_det: int = 1) -> dict:
    return {
        "frames": np.arange(n_frames, dtype=np.int32),
        "keypoints": np.array(
            [np.zeros((n_det, 17, 2), dtype=np.float32) for _ in range(n_frames)],
            dtype=object,
        ),
        "scores": np.array(
            [np.zeros((n_det, 17), dtype=np.float32) for _ in range(n_frames)],
            dtype=object,
        ),
    }


def test_loads_valid_artifact(tmp_path: Path) -> None:
    path = _write(tmp_path / "clip.npz", **_valid_payload())

    artifact = load_pose_artifact(path)

    assert len(artifact) == 3
    assert artifact.keypoints[0].shape == (1, 17, 2)


def test_accepts_frames_without_detections(tmp_path: Path) -> None:
    payload = _valid_payload(n_frames=2)
    payload["keypoints"] = np.array(
        [
            np.zeros((0, 17, 2), dtype=np.float32),
            np.zeros((1, 17, 2), dtype=np.float32),
        ],
        dtype=object,
    )
    payload["scores"] = np.array(
        [np.zeros((0, 17), dtype=np.float32), np.zeros((1, 17), dtype=np.float32)],
        dtype=object,
    )
    path = _write(tmp_path / "clip.npz", **payload)

    assert len(load_pose_artifact(path)) == 2


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PoseArtifactError, match="not found"):
        load_pose_artifact(tmp_path / "absent.npz")


@pytest.mark.parametrize("missing", ["frames", "keypoints", "scores"])
def test_rejects_missing_key(tmp_path: Path, missing: str) -> None:
    payload = _valid_payload()
    payload.pop(missing)
    path = _write(tmp_path / "clip.npz", **payload)

    with pytest.raises(PoseArtifactError, match=f"missing required keys: {missing}"):
        load_pose_artifact(path)


def test_rejects_non_dense_frames(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["frames"] = np.array([0, 2, 3], dtype=np.int32)
    path = _write(tmp_path / "clip.npz", **payload)

    with pytest.raises(PoseArtifactError, match="not dense and 0-based"):
        load_pose_artifact(path)


def test_rejects_wrong_frames_dtype(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["frames"] = np.arange(3, dtype=np.int64)
    path = _write(tmp_path / "clip.npz", **payload)

    with pytest.raises(PoseArtifactError, match="expected int32"):
        load_pose_artifact(path)


def test_rejects_length_mismatch(tmp_path: Path) -> None:
    payload = _valid_payload(n_frames=3)
    payload["frames"] = np.arange(4, dtype=np.int32)
    path = _write(tmp_path / "clip.npz", **payload)

    with pytest.raises(PoseArtifactError, match="length mismatch"):
        load_pose_artifact(path)


def test_rejects_wrong_keypoint_shape(tmp_path: Path) -> None:
    payload = _valid_payload(n_frames=2)
    payload["keypoints"] = np.array(
        [np.zeros((1, 13, 2), dtype=np.float32) for _ in range(2)], dtype=object
    )
    path = _write(tmp_path / "clip.npz", **payload)

    with pytest.raises(PoseArtifactError, match="expected trailing"):
        load_pose_artifact(path)


def test_rejects_detection_count_mismatch(tmp_path: Path) -> None:
    payload = _valid_payload(n_frames=2)
    payload["scores"] = np.array(
        [np.zeros((2, 17), dtype=np.float32) for _ in range(2)], dtype=object
    )
    path = _write(tmp_path / "clip.npz", **payload)

    with pytest.raises(PoseArtifactError, match="detection count mismatch"):
        load_pose_artifact(path)
