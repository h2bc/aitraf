"""VideoMAE evaluation pipeline for score prediction (regression)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mlflow
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from mlflow.data import from_huggingface
from mlflow import transformers as mlflow_transformers
from transformers import TrainingArguments, Trainer

from aitraf.metrics import (
    build_regression_metrics,
    compute_dummy_regression_preds,
    get_predicted_vs_actual_scatter_figure,
    get_residual_distribution_figure,
    get_residual_vs_predicted_scatter_figure,
    get_top_k_worst_errors,
)
from aitraf.processing.models.video_mae import process_sample
from aitraf.processing.utils import build_collate
from aitraf.logging import logger


@dataclass
class VideoMaeScorePredictionEvalCfg:
    """Configuration for evaluating VideoMAE score prediction."""

    model_uri: str
    manifests_dir: Path | str
    target_col: str
    clips_dir: Path | str
    batch_size: int
    num_workers: int
    sample_frames: int
    sampling_dist: str
    device: str
    output_dir: Path | str
    run_name: str
    experiment_name: str
    top_k_worst: int

    def __post_init__(self) -> None:
        self.manifests_dir = Path(self.manifests_dir)
        self.clips_dir = Path(self.clips_dir)
        self.output_dir = Path(self.output_dir)


def run_evaluation(config: VideoMaeScorePredictionEvalCfg) -> None:
    """Evaluate a fine-tuned VideoMAE regression model."""

    load_dotenv()

    dataset = load_dataset(
        "json",
        data_files={"test": str(config.manifests_dir / "test.jsonl")},
    )

    eval_dataset = dataset["test"]

    components = mlflow_transformers.load_model(
        config.model_uri, return_type="components"
    )
    model = components["model"].to(config.device)
    logger.info(
        f"VideoMAE evaluation running on device: {next(model.parameters()).device}"
    )
    processor = components["image_processor"]

    compute_metrics = build_regression_metrics()

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_eval_batch_size=config.batch_size,
        dataloader_num_workers=config.num_workers,
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
        target_column=config.target_col,
        label_transform=float,
    )

    data_collator = build_collate(process_fn)

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_input(from_huggingface(eval_dataset, name="test"), context="test")

        pred_logits, label_values, _ = trainer.predict(eval_dataset)

        predictions = np.asarray(pred_logits).squeeze()
        labels = np.asarray(label_values).squeeze()

        metrics = compute_metrics(predictions, labels)
        dummy_metrics = compute_metrics(
            compute_dummy_regression_preds(labels), labels
        )
        dummy_metrics = {f"dummy_{k}": v for k, v in dummy_metrics.items()}

        mlflow.log_metrics(metrics)
        mlflow.log_metrics(dummy_metrics)

        scatter_fig = get_predicted_vs_actual_scatter_figure(predictions, labels)
        mlflow.log_figure(scatter_fig, "predicted_vs_actual.png")

        residual_fig = get_residual_vs_predicted_scatter_figure(
            predictions, labels
        )
        mlflow.log_figure(residual_fig, "residuals_vs_predicted.png")
        
        residual_dist_fig = get_residual_distribution_figure(predictions, labels)
        mlflow.log_figure(residual_dist_fig, "residual_distribution.png")

        worst_misses = get_top_k_worst_errors(
            predictions,
            labels,
            eval_dataset.to_pandas(),
            top_k=config.top_k_worst,
        )

        if not worst_misses.empty:
            mlflow.log_table(worst_misses, "worst_misses.json")


__all__ = ["VideoMaeScorePredictionEvalCfg", "run_evaluation"]
