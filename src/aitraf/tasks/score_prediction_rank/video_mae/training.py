"""VideoMAE training pipeline for pairwise ranking."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow.data import from_pandas
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForVideoClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    VideoMAEImageProcessor,
)

from aitraf.datasets.score_prediction_rank import (
    ScorePredictionRankDataset,
    ScorePredictionRankSubset,
)
from aitraf.logging import logger
from aitraf.metrics import build_pairwise_metrics
from aitraf.models import PairwiseRanker
from aitraf.processing import load_target_label_mappings
from aitraf.processing.models.video_mae import process_pair_sample
from aitraf.processing.utils import build_collate


@dataclass
class VideoMaeScorePredictionRankTrainCfg:
    """Configuration for VideoMAE pairwise ranking training."""

    task_name: str
    model_name: str
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
    epochs: int
    experiment_name: str
    run_name: str
    freeze_backbone: bool
    early_stopping_patience: int
    max_train_samples: int | None = None

    def __post_init__(self) -> None:
        self.manifests_dir = Path(self.manifests_dir)
        self.vocab_path = Path(self.vocab_path)
        self.clips_dir = Path(self.clips_dir)
        self.output_dir = Path(self.output_dir)
        self.model_cache_dir = Path(self.model_cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)


def run_training(config: VideoMaeScorePredictionRankTrainCfg) -> str:
    """Train the VideoMAE pairwise ranker and log artifacts to MLflow."""

    load_dotenv()

    train_dataset = ScorePredictionRankDataset(
        manifests_dir=config.manifests_dir,
        split="train",
    )

    if config.max_train_samples is not None:
        max_count = min(config.max_train_samples, len(train_dataset))
        train_dataset = ScorePredictionRankSubset(train_dataset, range(max_count))

    val_dataset = ScorePredictionRankDataset(
        manifests_dir=config.manifests_dir,
        split="val",
    )

    if len(train_dataset) == 0:
        raise RuntimeError("No training pairs found for score_prediction_rank.")
    if len(val_dataset) == 0:
        raise RuntimeError("No validation pairs found for score_prediction_rank.")

    processor = VideoMAEImageProcessor.from_pretrained(
        config.backbone, cache_dir=str(config.model_cache_dir)
    )

    _, label2id, _ = load_target_label_mappings(config.vocab_path, "pair_label")

    model_config = AutoConfig.from_pretrained(
        config.backbone,
        cache_dir=str(config.model_cache_dir),
        trust_remote_code=True,
        num_labels=1,
        problem_type="regression",
    )

    scorer = AutoModelForVideoClassification.from_pretrained(
        config.backbone,
        config=model_config,
        cache_dir=str(config.model_cache_dir),
        trust_remote_code=True,
    ).to(config.device)

    logger.info(f"VideoMAE trainer using device: {next(scorer.parameters()).device}")

    if config.freeze_backbone:
        for param in scorer.base_model.parameters():
            param.requires_grad = False

    model = PairwiseRanker(
        scorer=scorer,
        loss_fn=nn.BCEWithLogitsLoss(),
    ).to(config.device)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        dataloader_num_workers=config.num_workers,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=["mlflow"],
        run_name=config.run_name,
        save_total_limit=1,
    )

    process_fn = partial(
        process_pair_sample,
        processor=processor,
        local_clips_dir=config.clips_dir,
        num_frames=config.sample_frames,
        sampling_dist=config.sampling_dist,
        label_transform=lambda label: label2id[str(label)],
    )

    data_collator = build_collate(process_fn)
    compute_metrics = build_pairwise_metrics()

    def trainer_compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
        pair_logits, labels = prediction
        pred_labels = (np.asarray(pair_logits).squeeze(-1) >= 0).astype(int)
        return compute_metrics(pred_labels, np.asarray(labels))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=trainer_compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        ],
    )

    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_input(
            from_pandas(pd.DataFrame(train_dataset.manifest_rows()), name="train"),
            context="training",
        )
        mlflow.log_input(
            from_pandas(pd.DataFrame(val_dataset.manifest_rows()), name="validation"),
            context="validation",
        )

        trainer.train()

        sample_pair = process_fn(train_dataset.manifest_rows()[0])
        model_info = mlflow.transformers.log_model(
            transformers_model={
                "model": model.scorer,
                "image_processor": processor,
            },
            name=f"{config.task_name}_{config.model_name}",
            input_example={
                "pixel_values": sample_pair["left_pixel_values"]
                .unsqueeze(0)
                .numpy()
                .tolist()
            },
        )

        return model_info.model_uri


__all__ = ["VideoMaeScorePredictionRankTrainCfg", "run_training"]
