"""Basic dataloading loop for score prediction rank experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from torch.utils.data import DataLoader

from aitraf.datasets.score_prediction_rank import ScorePredictionRankDataset


@dataclass
class VideoMaeScorePredictionRankTrainCfg:
    """Configuration for the current rank dataloading experiment."""

    manifests_dir: Path | str
    ranks_path: Path | str
    batch_size: int
    num_workers: int
    device: str
    output_dir: Path | str

    def __post_init__(self) -> None:
        self.manifests_dir = Path(self.manifests_dir)
        self.ranks_path = Path(self.ranks_path)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def run_training(config: VideoMaeScorePredictionRankTrainCfg) -> str:
    """Load the current rank dataset and inspect the first batch."""

    pin_memory = config.device != "cpu"

    train_dataset = ScorePredictionRankDataset(
        manifests_dir=config.manifests_dir,
        ranks_path=config.ranks_path,
        split="train",
    )

    print(f"dataset_length={len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=pin_memory,
        collate_fn=lambda x: x,
    )

    first_batch = next(iter(train_loader))

    print(json.dumps(first_batch, indent=2, ensure_ascii=False))


__all__ = ["VideoMaeScorePredictionRankTrainCfg", "run_training"]
