"""Basic dataloading loop for score prediction rank experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from aitraf.datasets.score_prediction_rank import ScorePredictionRankDataset


@dataclass
class VideoMaeScorePredictionRankTrainCfg:
    """Configuration for the current rank dataloading experiment."""

    manifests_dir: Path | str
    batch_size: int
    num_workers: int
    device: str
    output_dir: Path | str

    def __post_init__(self) -> None:
        self.manifests_dir = Path(self.manifests_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def run_training(config: VideoMaeScorePredictionRankTrainCfg) -> str:
    """Load the current rank dataset and inspect the first batch."""

    pin_memory = config.device != "cpu"

    train_dataset = ScorePredictionRankDataset(
        manifests_dir=config.manifests_dir,
        split="train",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=pin_memory,
        collate_fn=lambda x: x,
    )

    first_batch = next(iter(train_loader))

    print(f"batch_length={len(first_batch)}")
    print("video_ids:")
    for sample in first_batch:
        print(sample["video_id"])


__all__ = ["VideoMaeScorePredictionRankTrainCfg", "run_training"]
