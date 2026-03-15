"""Dataset for score-prediction rank experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class ScorePredictionRankDataset(Dataset):
    """Dataset that joins split manifests with pairwise rank annotations."""

    def __init__(
        self,
        *,
        manifests_dir: Path | str,
        ranks_path: Path | str,
        split: str,
    ) -> None:
        self.manifests_dir = Path(manifests_dir)
        self.ranks_path = Path(ranks_path)
        self.split = split
        manifest_rows = self._load_jsonl(self.manifests_dir / f"{split}.jsonl")
        rank_rows = self._load_jsonl(self.ranks_path)
        self.records = self._build_pair_records(manifest_rows, rank_rows)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self.records[index])

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")
        with path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    @classmethod
    def _build_pair_records(
        cls,
        manifest_rows: list[dict[str, Any]],
        rank_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        manifest_by_s3_path = {
            str(row["s3_path"]): row
            for row in manifest_rows
            if row.get("s3_path")
        }

        records: list[dict[str, Any]] = []
        for rank_row in rank_rows:
            left_value = rank_row.get("left")
            right_value = rank_row.get("right")

            if left_value is None or right_value is None:
                continue

            left_path = str(left_value)
            right_path = str(right_value)

            left_manifest = manifest_by_s3_path.get(left_path)
            right_manifest = manifest_by_s3_path.get(right_path)
            
            if left_manifest is None or right_manifest is None:
                continue

            pref = rank_row.get("pref") or {}
            records.append(
                {
                    "annotation_id": rank_row.get("annotation_id"),
                    "task_id": rank_row.get("task_id"),
                    "trick": rank_row.get("trick"),
                    "pair_label": pref.get("selected"),
                    "left_video_id": left_manifest["video_id"],
                    "right_video_id": right_manifest["video_id"],
                }
            )

        return records

    def manifest_rows(self) -> list[dict[str, Any]]:
        return list(self.records)


__all__ = ["ScorePredictionRankDataset"]
