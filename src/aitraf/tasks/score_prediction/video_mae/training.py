"""VideoMAE fine-tuning pipeline for score prediction (regression)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
from datasets import load_dataset
from dotenv import load_dotenv
from mlflow.data import from_huggingface
from transformers import (
    AutoConfig,
    AutoModelForVideoClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    VideoMAEImageProcessor,
)
from functools import partial

from aitraf.logging import logger
from aitraf.metrics import build_regression_metrics
from aitraf.processing.models.video_mae import process_sample
from aitraf.processing.utils import build_collate


@dataclass
class VideoMaeScorePredictionTrainCfg:
    """Configuration for VideoMAE score prediction training."""

    task_name: str
    model_name: str
    backbone: str
    manifests_dir: Path | str
    clips_dir: Path | str
    batch_size: int
    num_workers: int
    sample_frames: int
    sampling_dist: str
    device: str
    output_dir: Path | str
    epochs: int
    experiment_name: str
    run_name: str
    freeze_backbone: bool
    model_cache_dir: Path
    max_train_samples: int | None
    early_stopping_patience: int

    def __post_init__(self) -> None:
        self.manifests_dir = Path(self.manifests_dir)
        self.clips_dir = Path(self.clips_dir)
        self.output_dir = Path(self.output_dir)
        self.model_cache_dir = Path(self.model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)


def run_training(config: VideoMaeScorePredictionTrainCfg) -> str:
    """Train the VideoMAE regressor and log artifacts to MLflow."""

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

    processor = VideoMAEImageProcessor.from_pretrained(
        config.backbone, cache_dir=str(config.model_cache_dir)
    )

    model_config = AutoConfig.from_pretrained(
        config.backbone,
        cache_dir=str(config.model_cache_dir),
        trust_remote_code=True,
        num_labels=1,
        problem_type="regression",
        num_frames=config.sample_frames,
    )

    model = AutoModelForVideoClassification.from_pretrained(
        config.backbone,
        config=model_config,
        cache_dir=str(config.model_cache_dir),
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    ).to(config.device)

    logger.info(f"VideoMAE trainer using device: {next(model.parameters()).device}")

    if config.freeze_backbone:
        for param in model.base_model.parameters():
            param.requires_grad = False

    compute_metrics = build_regression_metrics()

    def trainer_compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
        pred_logits, actual_values = prediction

        pred_values = pred_logits.squeeze()
        label_values = actual_values.squeeze()

        return compute_metrics(pred_values, label_values)

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
        metric_for_best_model="mae",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=["mlflow"],
        run_name=config.run_name,
    )

    process_fn = partial(
        process_sample,
        processor=processor,
        local_clips_dir=config.clips_dir,
        num_frames=config.sample_frames,
        sampling_dist=config.sampling_dist,
        label_key="execution_score",
    )

    data_collator = build_collate(process_fn)

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
        mlflow.log_param("frozen", config.freeze_backbone)
        mlflow.log_param("train_sampling_dist", config.sampling_dist)
        mlflow.log_input(
            from_huggingface(dataset["train"], name="train"), context="training"
        )
        mlflow.log_input(
            from_huggingface(dataset["validation"], name="validation"),
            context="validation",
        )

        trainer.train()

        sample_clip = process_fn(dataset["train"][0])

        model_info = mlflow.transformers.log_model(
            transformers_model={
                "model": model,
                "image_processor": processor,
            },
            name=f"{config.task_name}_{config.model_name}",
            input_example={
                "pixel_values": sample_clip["pixel_values"]
                .unsqueeze(0)
                .numpy()
                .tolist()
            },
        )

        return model_info.model_uri


__all__ = ["VideoMaeScorePredictionTrainCfg", "run_training"]
