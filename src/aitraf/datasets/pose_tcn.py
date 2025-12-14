"""Dataset for pose-based TCN experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from torch.utils.data import Dataset

from aitraf.data_ops import schema
from torch.utils.data import Subset


class PoseTCNSubset(Subset):
    def __init__(self, dataset: PoseTCNDataset, indices: Sequence[int]) -> None:
        if not isinstance(dataset, PoseTCNDataset):
            raise TypeError("PoseTCNSubset requires a PoseTCNDataset base.")
        super().__init__(dataset, indices)

    def manifest_rows(self) -> list[dict]:
        base_rows = self.dataset.manifest_rows()
        return [base_rows[idx] for idx in self.indices]


class PoseTCNDataset(Dataset):
    """Lightweight dataset that returns pose payloads per clip."""

    def __init__(
        self, manifests_dir: Path | str, poses_dir: Path | str, split: str
    ) -> None:
        self.manifests_dir = Path(manifests_dir)
        self.poses_dir = Path(poses_dir)
        self.split = split
        self.records = self._load_manifest(self.manifests_dir / f"{split}.jsonl")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        pose_path = self._resolve_pose_path(row["video_id"])

        payload = np.load(pose_path, allow_pickle=True)

        return {
            "video_id": row["video_id"],
            "keypoints": payload["keypoints"],
            "scores": payload["scores"],
            "frames": payload["frames"],
            "label": row[schema.TARGET_COLUMN],
        }

    def _resolve_pose_path(self, video_id: str) -> Path:
        pose_stem = Path(video_id).stem
        pose_path = self.poses_dir / f"{pose_stem}.npz"

        if not pose_path.exists():
            raise FileNotFoundError(f"Pose file not found: {pose_path}")
        return pose_path

    @staticmethod
    def _load_manifest(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        with path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def manifest_rows(self) -> list[dict]:
        return list(self.records)


__all__ = ["PoseTCNDataset", "PoseTCNSubset"]
