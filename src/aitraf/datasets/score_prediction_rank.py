"""Dataset for score-prediction rank experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class ScorePredictionRankDataset(Dataset):
    """Placeholder dataset that currently returns manifest rows per split."""

    def __init__(self, *, manifests_dir: Path | str, split: str) -> None:
        self.manifests_dir = Path(manifests_dir)
        self.split = split
        self.records = self._load_manifest(self.manifests_dir / f"{split}.jsonl")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self.records[index])

    @staticmethod
    def _load_manifest(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        with path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def manifest_rows(self) -> list[dict[str, Any]]:
        return list(self.records)


__all__ = ["ScorePredictionRankDataset"]
