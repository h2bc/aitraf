"""VideoMAE temporal-fusion training for trick classification."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mlflow
import mlflow.pytorch
from datasets import load_dataset
from dotenv import load_dotenv
from mlflow.data import from_huggingface
from torch import nn
from transformers import (
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

from aitraf_train.logging import logger
from aitraf_train.metrics import accuracy, calc_metrics, compute_pred_ids, f1_macro
from aitraf_core.pre_processing.paths import video_feature_cache_dir
from aitraf_train.data.labels import (
    build_class_weights,
    build_label_transform,
    load_target_label_mappings,
)
from aitraf_core.processing.models.video_mae_temporal_fusion import (
    VideoMaeTemporalFusionClassifier,
    process_temporal_fusion_feature_sample,
)
from aitraf_train.data.collate import build_collate


@dataclass
class VideoMaeTemporalFusionTrickClassificationTrainCfg:
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
    early_stopping_patience: int | None
    fusion_layers: int
    fusion_heads: int
    fusion_queries: int
    query_init_std: float
    fusion_dropout: float
    loss: str
    use_class_weights: bool
    best_model_metric: str
    seed: int

    def __post_init__(self) -> None:
        self.manifests_dir = Path(self.manifests_dir)
        self.vocab_path = Path(self.vocab_path)
        self.features_dir = Path(self.features_dir)
        self.output_dir = Path(self.output_dir)


def run_training(config: VideoMaeTemporalFusionTrickClassificationTrainCfg) -> str:
    load_dotenv()
    set_seed(config.seed)

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

    label_mappings = load_target_label_mappings(config.vocab_path, "trick")
    label_transform = build_label_transform(label_mappings.label2id)
    feature_cache_dir = video_feature_cache_dir(
        features_dir=config.features_dir,
        backbone=config.backbone,
        num_clips=config.num_clips,
        sample_frames=config.sample_frames,
        sampling_dist=config.sampling_dist,
    )
    process_fn = partial(
        process_temporal_fusion_feature_sample,
        feature_cache_dir=feature_cache_dir,
        label_key="trick",
        label_transform=label_transform,
    )
    sample_clip = process_fn(dataset["train"][0])
    class_weights = (
        build_class_weights(
            [label_transform(row["trick"]) for row in dataset["train"]],
            num_labels=len(label_mappings.labels),
            device=config.device,
        )
        if config.use_class_weights
        else None
    )
    model = VideoMaeTemporalFusionClassifier(
        hidden_size=sample_clip["features"].shape[-1],
        num_labels=len(label_mappings.labels),
        num_clips=config.num_clips,
        num_queries=config.fusion_queries,
        fusion_layers=config.fusion_layers,
        fusion_heads=config.fusion_heads,
        fusion_dropout=config.fusion_dropout,
        query_init_std=config.query_init_std,
        loss_fn=nn.CrossEntropyLoss(weight=class_weights),
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
            (accuracy, f1_macro),
        )

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        dataloader_num_workers=config.num_workers,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        seed=config.seed,
        data_seed=config.seed,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
        report_to=["mlflow"],
        run_name=config.run_name,
    )
    callbacks = []
    if config.early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=trainer_compute_metrics,
        callbacks=callbacks,
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
            metadata=label_mappings.model_metadata(),
        )
        return model_info.model_uri


def _training_params(
    config: VideoMaeTemporalFusionTrickClassificationTrainCfg,
) -> dict[str, object]:
    return {
        "backbone": config.backbone,
        "frozen": True,
        "train_sampling_dist": config.sampling_dist,
        "num_frames": config.sample_frames,
        "num_clips": config.num_clips,
        "fusion_layers": config.fusion_layers,
        "fusion_heads": config.fusion_heads,
        "fusion_queries": config.fusion_queries,
        "query_init_std": config.query_init_std,
        "fusion_dropout": config.fusion_dropout,
        "loss": config.loss,
        "use_class_weights": config.use_class_weights,
        "best_model_metric": config.best_model_metric,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "max_epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "metric_for_best_model": "accuracy",
    }


__all__ = ["VideoMaeTemporalFusionTrickClassificationTrainCfg", "run_training"]
