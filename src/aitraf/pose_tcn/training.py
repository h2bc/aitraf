"""Pose TCN preview training loop"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from aitraf.logging import logger
from aitraf.pose_tcn.data import PoseTCNDataset


@dataclass
class PoseTCNTrainingConfig:
    """Configuration for the pose TCN data inspection loop."""

    manifests_dir: Path | str
    poses_dir: Path | str
    batch_size: int
    num_workers: int

    def __post_init__(self) -> None:
        self.manifests_dir = Path(self.manifests_dir)
        self.poses_dir = Path(self.poses_dir)


def run_training(config: PoseTCNTrainingConfig) -> None:
    """Load the dataset and iterate through a few batches, logging metadata."""

    _preview_split(
        split_name="train",
        manifests_dir=config.manifests_dir,
        poses_dir=config.poses_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    _preview_split(
        split_name="val",
        manifests_dir=config.manifests_dir,
        poses_dir=config.poses_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )


def _preview_split(
    *,
    split_name: str,
    manifests_dir: Path,
    poses_dir: Path,
    batch_size: int,
    num_workers: int,
) -> None:
    dataset = PoseTCNDataset(
        manifests_dir=manifests_dir,
        poses_dir=poses_dir,
        split=split_name,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )

    logger.info(
        "Inspecting pose dataset split={} batch_size={} preview_batches={}",
        split_name,
        batch_size,
        1,
    )

    for batch_idx, batch in enumerate(dataloader):
        first = batch[0]
        logger.info(
            "Split={} batch={} size={} first_video={} keys={}",
            split_name,
            batch_idx,
            len(batch),
            first["video_id"],
            list(first.keys()),
        )
        break


__all__ = ["PoseTCNTrainingConfig", "run_training"]
