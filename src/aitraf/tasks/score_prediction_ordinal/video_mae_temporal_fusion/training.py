"""VideoMAE temporal-fusion training for ordinal score prediction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mlflow
import mlflow.pytorch
from datasets import load_dataset
from dlordinal.losses import EMDLoss
from dotenv import load_dotenv
from mlflow.data import from_huggingface
from transformers import EarlyStoppingCallback, EvalPrediction, Trainer, TrainingArguments

from aitraf.logging import logger
from aitraf.metrics import calc_metrics, compute_pred_ids
from aitraf.processing import load_target_label_mappings
from aitraf.processing.models.video_mae_temporal_fusion import (
    VideoMaeTemporalFusionClassifier,
    process_temporal_fusion_feature_sample,
)
from aitraf.processing.utils import build_collate
from aitraf.tasks.score_prediction_ordinal.metrics import amae, mae


@dataclass
class VideoMaeTemporalFusionScorePredictionOrdinalTrainCfg:
    task_name: str
    model_name: str
    backbone: str
    manifests_dir: Path | str
    vocab_path: Path | str
    features_dir: Path | str
    batch_size: int
    num_workers: int
    sample_frames: int
    num_clips: int
    sampling_dist: str
    device: str
    output_dir: Path | str
    epochs: int
    learning_rate: float
    experiment_name: str
    run_name: str
    max_train_samples: int | None
    early_stopping_patience: int
    fusion_layers: int
    fusion_heads: int
    fusion_dropout: float

    def __post_init__(self) -> None:
        self.manifests_dir = Path(self.manifests_dir)
        self.vocab_path = Path(self.vocab_path)
        self.features_dir = Path(self.features_dir)
        self.output_dir = Path(self.output_dir)


def run_training(config: VideoMaeTemporalFusionScorePredictionOrdinalTrainCfg) -> str:
    load_dotenv()

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(config.manifests_dir / "train.jsonl"),
            "validation": str(config.manifests_dir / "val.jsonl"),
        },
    )
    if config.max_train_samples is not None:
        dataset["train"] = dataset["train"].select(
            range(min(config.max_train_samples, len(dataset["train"])))
        )

    labels, label2id, _ = load_target_label_mappings(
        config.vocab_path, "execution_score"
    )
    process_fn = partial(
        process_temporal_fusion_feature_sample,
        features_dir=config.features_dir,
        backbone=config.backbone,
        num_frames=config.sample_frames,
        num_clips=config.num_clips,
        sampling_dist=config.sampling_dist,
        label_key="execution_score",
        label_transform=lambda label: label2id[str(label)],
    )
    sample_clip = process_fn(dataset["train"][0])
    model = _build_model(
        config,
        num_labels=len(labels),
        hidden_size=sample_clip["features"].shape[-1],
    ).to(config.device)
    logger.info(
        f"VideoMAE temporal-fusion trainer using device: {next(model.parameters()).device}"
    )
    data_collator = build_collate(process_fn)

    def trainer_compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
        pred_logits, actual_ids = prediction
        return calc_metrics(
            compute_pred_ids(pred_logits),
            actual_ids,
            (amae, mae),
        )

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        dataloader_num_workers=config.num_workers,
        dataloader_persistent_workers=config.num_workers > 0,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="amae",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=["mlflow"],
        run_name=config.run_name,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
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
        mlflow.log_params(_training_params(config))
        mlflow.log_input(
            from_huggingface(dataset["train"], name="train"), context="training"
        )
        mlflow.log_input(
            from_huggingface(dataset["validation"], name="validation"),
            context="validation",
        )
        trainer.train()

        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name=f"{config.task_name}_{config.model_name}",
            input_example={"features": sample_clip["features"].unsqueeze(0).numpy()},
        )
        return model_info.model_uri


def _build_model(
    config: VideoMaeTemporalFusionScorePredictionOrdinalTrainCfg,
    *,
    num_labels: int,
    hidden_size: int,
) -> VideoMaeTemporalFusionClassifier:
    return VideoMaeTemporalFusionClassifier(
        hidden_size=hidden_size,
        num_labels=num_labels,
        num_clips=config.num_clips,
        fusion_layers=config.fusion_layers,
        fusion_heads=config.fusion_heads,
        fusion_dropout=config.fusion_dropout,
        loss_fn=EMDLoss(num_classes=num_labels),
    )


def _training_params(
    config: VideoMaeTemporalFusionScorePredictionOrdinalTrainCfg,
) -> dict[str, object]:
    return {
        "backbone": config.backbone,
        "frozen": True,
        "train_sampling_dist": config.sampling_dist,
        "num_frames": config.sample_frames,
        "num_clips": config.num_clips,
        "fusion_layers": config.fusion_layers,
        "fusion_heads": config.fusion_heads,
        "fusion_dropout": config.fusion_dropout,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "max_epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "metric_for_best_model": "amae",
    }


__all__ = ["VideoMaeTemporalFusionScorePredictionOrdinalTrainCfg", "run_training"]
