"""Basic dataloading loop for score prediction rank experiments."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

from transformers import VideoMAEImageProcessor
from torch.utils.data import DataLoader

from aitraf.datasets.score_prediction_rank import ScorePredictionRankDataset
from aitraf.processing.models.video_mae import process_pair_sample
from aitraf.processing.utils import build_collate




@dataclass
class VideoMaeScorePredictionRankTrainCfg:
    """Configuration for the current rank dataloading experiment."""

    backbone: str
    manifests_dir: Path | str
    ranks_path: Path | str
    clips_dir: Path | str
    batch_size: int
    num_workers: int
    sample_frames: int
    sampling_dist: str
    device: str
    output_dir: Path | str
    model_cache_dir: Path | str

    def __post_init__(self) -> None:
        self.manifests_dir = Path(self.manifests_dir)
        self.ranks_path = Path(self.ranks_path)
        self.clips_dir = Path(self.clips_dir)
        self.output_dir = Path(self.output_dir)
        self.model_cache_dir = Path(self.model_cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)


def run_training(config: VideoMaeScorePredictionRankTrainCfg) -> str:
    """Load the current rank dataset and inspect the first processed batch."""

    pin_memory = config.device != "cpu"

    processor = VideoMAEImageProcessor.from_pretrained(
        config.backbone, cache_dir=str(config.model_cache_dir)
    )

    def encode_pair_label(label: str) -> int:
        if label == "left":
            return 0
        if label == "right":
            return 1
        raise ValueError(f"Unsupported pair label: {label}")


    process_fn = partial(
        process_pair_sample,
        processor=processor,
        local_clips_dir=config.clips_dir,
        num_frames=config.sample_frames,
        sampling_dist=config.sampling_dist,
        label_transform=encode_pair_label,
    )
    collate_fn = build_collate(process_fn)

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
        collate_fn=collate_fn,
    )

    first_batch = next(iter(train_loader))

    print(f"batch_keys={list(first_batch.keys())}")
    print(f"left_pixel_values_shape={tuple(first_batch['left_pixel_values'].shape)}")
    print(f"right_pixel_values_shape={tuple(first_batch['right_pixel_values'].shape)}")
    print(f"labels_shape={tuple(first_batch['labels'].shape)}")
    print(f"labels={first_batch['labels'].tolist()}")


__all__ = ["VideoMaeScorePredictionRankTrainCfg", "run_training"]
