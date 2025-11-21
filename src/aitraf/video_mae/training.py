"""VideoMAE data-loading scaffolding used by the CLI entrypoint."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from functools import reduce
import torch
import evaluate


from transformers import (
    AutoConfig,
    AutoModelForVideoClassification,
    Trainer,
    TrainingArguments,
    VideoMAEImageProcessor,
)

from aitraf.video_mae.processing import process_clip
import json

from datasets import load_dataset

from aitraf.data import schema
from dotenv import load_dotenv
import mlflow

import numpy as np


@dataclass
class VideoMAETrainingConfig:
    """Minimal configuration for the current skeleton run."""

    backbone: str
    manifests_dir: Path | str
    clips_dir: Path | str
    batch_size: int
    num_workers: int
    sample_frames: int
    device: str
    output_dir: Path | str
    epochs: int
    experiment_name: str
    run_name: str

    def __post_init__(self) -> None:
        self.manifests_dir = Path(self.manifests_dir)
        self.clips_dir = Path(self.clips_dir)
        self.output_dir = Path(self.output_dir)


def run_training(config: VideoMAETrainingConfig) -> None:
    """Load a couple of batches and print their contents (no training yet)."""

    load_dotenv()
    mlflow.set_experiment(config.experiment_name)

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(config.manifests_dir / "train.jsonl"),
            "validation": str(config.manifests_dir / "val.jsonl"),
        },
    )

    labels_config = _read_json(config.manifests_dir / "labels.json")
    label2id = labels_config[schema.TARGET_COLUMN]["label2id"]
    id2label = labels_config[schema.TARGET_COLUMN]["id2label"]

    processor = VideoMAEImageProcessor.from_pretrained(config.backbone)
    model_config = AutoConfig.from_pretrained(
        config.backbone,
        trust_remote_code=True,
        label2id=label2id,
        id2label=id2label,
        num_labels=len(label2id),
    )

    model = AutoModelForVideoClassification.from_pretrained(
        config.backbone, config=model_config, trust_remote_code=True
    ).to(config.device)

    accuracy_metric = evaluate.load("accuracy")
    weighted_metrics = evaluate.combine(["precision", "recall", "f1"])

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        accuracy = accuracy_metric.compute(
            predictions=preds,
            references=labels,
        )["accuracy"]

        weighted = weighted_metrics.compute(
            predictions=preds,
            references=labels,
            average="weighted",
        )

        return {
            "accuracy": accuracy,
            "precision_weighted": weighted["precision"],
            "recall_weighted": weighted["recall"],
            "f1_weighted": weighted["f1"],
        }

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        dataloader_num_workers=config.num_workers,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,
        report_to=["mlflow"],
        run_name=config.run_name,
    )

    def _collate(batch):
        # [{pixels, label}, {pixels, label} .. ]
        processed_batch = [
            process_clip(
                row, processor, config.clips_dir, label2id, config.sample_frames
            )
            for row in batch
        ]

        # {pixels: [], labels: []}
        pivot = reduce(
            lambda acc, x: {k: acc.get(k, []) + [x[k]] for k in x}, processed_batch, {}
        )

        return {k: torch.stack(v) for k, v in pivot.items()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=_collate,
        compute_metrics=_compute_metrics,
    )

    trainer.train()


def _read_json(json_path: Path) -> dict[str, Any]:
    with open(json_path) as f:
        data = json.load(f)

    return data
