"""VideoMAE evaluation pipeline for binary score prediction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mlflow
from datasets import load_dataset
from dotenv import load_dotenv
from mlflow import transformers as mlflow_transformers
from mlflow.data import from_huggingface
from transformers import Trainer, TrainingArguments

from aitraf.logging import logger
from aitraf.metrics import (
    build_classification_metrics,
    compute_dummy_classification_pred_ids,
    compute_pred_ids,
    get_confusion_matrix_figure,
    get_per_class_f1_figure,
    get_target_distribution_figure,
    get_top_k_worst_misses,
)
from aitraf.processing import load_target_label_mappings
from aitraf.processing.models.video_mae import process_sample
from aitraf.processing.utils import build_collate


@dataclass
class VideoMaeScorePredictionBinaryEvalCfg:
    """Configuration for evaluating VideoMAE binary score prediction."""

    model_uri: str
    manifests_dir: Path | str
    vocab_path: Path | str
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
        self.vocab_path = Path(self.vocab_path)
        self.clips_dir = Path(self.clips_dir)
        self.output_dir = Path(self.output_dir)


def run_evaluation(config: VideoMaeScorePredictionBinaryEvalCfg) -> None:
    """Evaluate a fine-tuned VideoMAE binary classifier."""

    load_dotenv()

    dataset = load_dataset(
        "json",
        data_files={"test": str(config.manifests_dir / "test.jsonl")},
    )

    eval_dataset = dataset["test"]

    label_names, label2id, id2label = load_target_label_mappings(
        config.vocab_path, "quality_label"
    )

    components = mlflow_transformers.load_model(
        config.model_uri, return_type="components"
    )
    model = components["model"].to(config.device)
    logger.info(
        f"VideoMAE evaluation running on device: {next(model.parameters()).device}"
    )
    processor = components["image_processor"]

    compute_metrics = build_classification_metrics()

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
        label_key="quality_label",
        label_transform=lambda label: label2id[str(label)],
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

        pred_logits, label_ids, _ = trainer.predict(eval_dataset)

        pred_ids = compute_pred_ids(pred_logits)
        metrics = compute_metrics(pred_ids, label_ids)
        mlflow.log_metrics(metrics)

        dummy_pred_ids = compute_dummy_classification_pred_ids(label_ids)
        dummy_metrics = compute_metrics(dummy_pred_ids, label_ids)
        mlflow.log_metrics({f"dummy_{k}": v for k, v in dummy_metrics.items()})

        dist_fig = get_target_distribution_figure(
            pred_ids,
            label_ids,
            label_names,
            id2label,
        )
        mlflow.log_figure(dist_fig, "predicted_vs_actual_target_counts.png")

        cm_fig = get_confusion_matrix_figure(pred_ids, label_ids, label_names)
        mlflow.log_figure(cm_fig, "confusion_matrix.png")

        f1_fig = get_per_class_f1_figure(pred_ids, label_ids, label_names)
        mlflow.log_figure(f1_fig, "per_class_f1.png")

        worst_misses = get_top_k_worst_misses(
            pred_logits,
            label_ids,
            eval_dataset.to_pandas(),
            id2label,
            top_k=config.top_k_worst,
        )

        if not worst_misses.empty:
            mlflow.log_table(worst_misses, "worst_misses.json")


__all__ = ["VideoMaeScorePredictionBinaryEvalCfg", "run_evaluation"]
