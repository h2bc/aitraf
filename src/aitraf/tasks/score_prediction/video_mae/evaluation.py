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
    EvalModel,
    EvalSet,
    build_training_params,
    calc_metrics_for_models,
    compute_dummy_regression_preds,
    flatten_metrics_report,
    get_predicted_vs_actual_scatter_figure,
    get_residual_distribution_figure,
    get_residual_vs_predicted_scatter_figure,
    get_top_k_worst_errors,
    mae,
    metrics_to_df,
    params_to_df,
    r2,
    rmse,
)
from aitraf.processing.models.video_mae import process_sample
from aitraf.processing.utils import build_collate
from aitraf.logging import logger


@dataclass
class VideoMaeScorePredictionEvalCfg:
    """Configuration for evaluating VideoMAE score prediction."""

    model_uri: str
    manifests_dir: Path | str
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
        data_files={
            "train": str(config.manifests_dir / "train.jsonl"),
            "test": str(config.manifests_dir / "test.jsonl"),
        },
    )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    components = mlflow_transformers.load_model(
        config.model_uri, return_type="components"
    )
    model = components["model"].to(config.device)
    logger.info(
        f"VideoMAE evaluation running on device: {next(model.parameters()).device}"
    )
    processor = components["image_processor"]

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
        label_key="execution_score",
        label_transform=float,
    )

    data_collator = build_collate(process_fn)

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    source_train_run_id = mlflow.models.get_model_info(config.model_uri).run_id
    source_train_params = build_training_params(source_train_run_id) | {
        "sampling_dist": config.sampling_dist
    }

    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params(source_train_params)
        params_df = params_to_df(source_train_params)
        mlflow.log_table(params_df, "params_table.json")
        mlflow.log_input(from_huggingface(train_dataset, name="train"), context="train")
        mlflow.log_input(from_huggingface(test_dataset, name="test"), context="test")

        train_pred_logits, train_label_values, _ = trainer.predict(train_dataset)
        train_predictions = np.asarray(train_pred_logits).reshape(-1)
        train_labels = np.asarray(train_label_values).reshape(-1)

        test_pred_logits, test_label_values, _ = trainer.predict(test_dataset)
        test_predictions = np.asarray(test_pred_logits).reshape(-1)
        test_labels = np.asarray(test_label_values).reshape(-1)

        metrics_report = calc_metrics_for_models(
            eval_models=[
                EvalModel(
                    name="dummy",
                    sets=[
                        EvalSet(
                            name="train",
                            predictions=compute_dummy_regression_preds(train_labels),
                            labels=train_labels,
                        ),
                        EvalSet(
                            name="test",
                            predictions=compute_dummy_regression_preds(test_labels),
                            labels=test_labels,
                        ),
                    ],
                ),
                EvalModel(
                    name="video_mae",
                    sets=[
                        EvalSet(
                            name="train",
                            predictions=train_predictions,
                            labels=train_labels,
                        ),
                        EvalSet(
                            name="test",
                            predictions=test_predictions,
                            labels=test_labels,
                        ),
                    ],
                ),
            ],
            eval_metrics=(
                mae,
                rmse,
                r2,
            ),
        )
        all_metrics = flatten_metrics_report(metrics_report)

        mlflow.log_metrics(all_metrics)
        metrics_df = metrics_to_df(metrics_report)
        mlflow.log_table(metrics_df, "metrics_table.json")

        scatter_fig = get_predicted_vs_actual_scatter_figure(test_predictions, test_labels)
        mlflow.log_figure(scatter_fig, "predicted_vs_actual.png")

        residual_fig = get_residual_vs_predicted_scatter_figure(
            test_predictions, test_labels
        )
        mlflow.log_figure(residual_fig, "residuals_vs_predicted.png")

        residual_dist_fig = get_residual_distribution_figure(
            test_predictions, test_labels
        )
        mlflow.log_figure(residual_dist_fig, "residual_distribution.png")

        worst_misses = get_top_k_worst_errors(
            test_predictions,
            test_labels,
            test_dataset.to_pandas(),
            top_k=config.top_k_worst,
        )

        if not worst_misses.empty:
            mlflow.log_table(worst_misses, "worst_misses.json")


__all__ = ["VideoMaeScorePredictionEvalCfg", "run_evaluation"]
