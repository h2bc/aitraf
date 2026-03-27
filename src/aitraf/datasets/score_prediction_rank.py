"""Dataset for score-prediction rank experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from torch.utils.data import Dataset, Subset


class ScorePredictionRankSubset(Subset):
    def __init__(
        self, dataset: ScorePredictionRankDataset, indices: Sequence[int]
    ) -> None:
        if not isinstance(dataset, ScorePredictionRankDataset):
            raise TypeError(
                "ScorePredictionRankSubset requires a ScorePredictionRankDataset base."
            )
        super().__init__(dataset, indices)

    def manifest_rows(self) -> list[dict]:
        base_rows = self.dataset.manifest_rows()
        return [base_rows[idx] for idx in self.indices]


class ScorePredictionRankDataset(Dataset):
    """Dataset backed directly by pairwise manifest rows."""

    def __init__(
        self,
        *,
        manifests_dir: Path | str,
        split: str,
    ) -> None:
        self.manifests_dir = Path(manifests_dir)
        self.split = split
        self.records = self._load_jsonl(self.manifests_dir / f"{split}.jsonl")

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

    def manifest_rows(self) -> list[dict[str, Any]]:
        return list(self.records)

__all__ = ["ScorePredictionRankDataset", "ScorePredictionRankSubset"]
