"""Basic dataloading loop for score prediction rank experiments."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

from transformers import VideoMAEImageProcessor
from torch.utils.data import DataLoader

from aitraf.datasets.score_prediction_rank import ScorePredictionRankDataset
from aitraf.processing import load_target_label_mappings
from aitraf.processing.models.video_mae import process_pair_sample
from aitraf.processing.utils import build_collate


@dataclass
class VideoMaeScorePredictionRankTrainCfg:
    """Configuration for the current rank dataloading experiment."""

    backbone: str
    manifests_dir: Path | str
    vocab_path: Path | str
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
        self.vocab_path = Path(self.vocab_path)
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
    _, label2id, _ = load_target_label_mappings(config.vocab_path, "pair_label")

    process_fn = partial(
        process_pair_sample,
        processor=processor,
        local_clips_dir=config.clips_dir,
        num_frames=config.sample_frames,
        sampling_dist=config.sampling_dist,
        label_transform=lambda label: label2id[str(label)],
    )
    collate_fn = build_collate(process_fn)

    train_dataset = ScorePredictionRankDataset(
        manifests_dir=config.manifests_dir,
        split="train",
    )
    val_dataset = ScorePredictionRankDataset(
        manifests_dir=config.manifests_dir,
        split="val",
    )

    print(f"train_dataset_length={len(train_dataset)}")
    print(f"val_dataset_length={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    first_train_batch = next(iter(train_loader))
    first_val_batch = next(iter(val_loader))

    print(f"train_batch_keys={list(first_train_batch.keys())}")
    print(
        f"train_left_pixel_values_shape={tuple(first_train_batch['left_pixel_values'].shape)}"
    )
    print(
        f"train_right_pixel_values_shape={tuple(first_train_batch['right_pixel_values'].shape)}"
    )
    print(f"train_labels_shape={tuple(first_train_batch['labels'].shape)}")
    print(f"train_labels={first_train_batch['labels'].tolist()}")

    print(f"val_batch_keys={list(first_val_batch.keys())}")
    print(
        f"val_left_pixel_values_shape={tuple(first_val_batch['left_pixel_values'].shape)}"
    )
    print(
        f"val_right_pixel_values_shape={tuple(first_val_batch['right_pixel_values'].shape)}"
    )
    print(f"val_labels_shape={tuple(first_val_batch['labels'].shape)}")
    print(f"val_labels={first_val_batch['labels'].tolist()}")


__all__ = ["VideoMaeScorePredictionRankTrainCfg", "run_training"]
