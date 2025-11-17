"""VideoMAE data-loading scaffolding used by the CLI entrypoint."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForVideoClassification,
    Trainer,
    TrainingArguments,
    VideoMAEImageProcessor,
)

from aitraf.logging import logger
from aitraf.video_mae.processing import load_clip


@dataclass
class VideoMAETrainingConfig:
    """Minimal configuration for the current skeleton run."""

    backbone: str
    manifest_path: Path | str
    clips_dir: Path | str
    batch_size: int = 2
    num_workers: int = 4
    num_frames: int = 16
    device: str = "cuda"
    output_dir: Path | str = "runs/video_mae"
    max_batches: int = 2

    def __post_init__(self) -> None:
        self.manifest_path = Path(self.manifest_path)
        self.clips_dir = Path(self.clips_dir)
        self.output_dir = Path(self.output_dir)


def run_training(config: VideoMAETrainingConfig) -> None:
    """Load a couple of batches and print their contents (no training yet)."""

    processor = VideoMAEImageProcessor.from_pretrained(config.backbone)
    model_config = AutoConfig.from_pretrained(config.backbone, trust_remote_code=True)
    model = AutoModelForVideoClassification.from_pretrained(
        config.backbone, config=model_config, trust_remote_code=True
    ).to(config.device)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=1,
    )

    trainer = Trainer(model=model, args=training_args)

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


def _collate(batch, *, processor, clips_dir, num_frames):
    processed = []
    for row in batch:
        try:
            item = load_clip(
                row=row,
                processor=processor,
                clips_dir=clips_dir,
                num_frames=num_frames,
            )
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("Skipping row due to {}", exc)
            continue
        processed.append(item)
    return processed
