"""VideoMAE data-loading scaffolding used by the CLI entrypoint."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import DataLoader
from transformers import VideoMAEImageProcessor

from aitraf.logging import logger
from aitraf.video_mae.processing import load_clip


@dataclass
class VideoMAETrainingConfig:
    """Minimal configuration for the current skeleton run."""

    backbone: str
    manifest_path: Path | str
    clips_dir: Path | str
    batch_size: int = 2
    num_workers: int = 0
    num_frames: int = 16
    label_column: str = "trick"
    max_batches: int = 2

    def __post_init__(self) -> None:
        self.manifest_path = Path(self.manifest_path)
        self.clips_dir = Path(self.clips_dir)


def run_training(config: VideoMAETrainingConfig) -> None:
    """Load a couple of batches and print their contents (no training yet)."""

    processor = VideoMAEImageProcessor.from_pretrained(config.backbone)
    dataset = _read_dataset(config.manifest_path)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=lambda rows: _collate(
            rows,
            processor=processor,
            clips_dir=config.clips_dir,
            num_frames=config.num_frames,
            label_column=config.label_column,
            device=config.device,
        ),
    )

    logger.info("Inspecting batches from {}", config.manifest_path)
    for idx, batch in enumerate(loader):
        if not batch:
            continue
        logger.info("Batch {}: {} clips processed", idx, len(batch))
        for item in batch:
            logger.info("  clip={} label={}", item["clip_path"], item["label"])
        if idx + 1 >= config.max_batches:
            break


def _read_dataset(manifest_path: Path) -> list[dict[str, Any]]:
    df = pd.read_json(manifest_path, lines=True)
    return df.to_dict(orient="records")


def _collate(batch, *, processor, clips_dir, num_frames, label_column, device):
    processed = []
    for row in batch:
        try:
            item = load_clip(
                row=row,
                processor=processor,
                clips_dir=clips_dir,
                num_frames=num_frames,
                label_column=label_column,
                device=device,
            )
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("Skipping row due to {}", exc)
            continue
        processed.append(item)
    return processed
