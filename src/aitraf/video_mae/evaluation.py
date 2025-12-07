"""VideoMAE evaluation pipeline."""

from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from mlflow.data import from_huggingface
import mlflow
from mlflow import transformers as mlflow_transformers
from transformers import (
    TrainingArguments,
    Trainer,
)

from aitraf.video_mae.utils import build_collate
from aitraf.processing import load_target_label_mappings
from aitraf.metrics import (
    build_compute_metrics,
    get_confusion_matrix_figure,
    get_target_distribution_figure,
    get_per_class_f1_figure,
    compute_pred_ids,
    compute_dummy_pred_ids,
    get_top_k_worst_misses,
)


@dataclass
class VideoMAEEvalConfig:
    """Configuration for VideoMAE evaluation."""

    backbone: str
    model_uri: str
    manifests_dir: Path | str
    clips_dir: Path | str
    batch_size: int
    num_workers: int
    sample_frames: int
    device: str
    output_dir: Path | str
    run_name: str
    experiment_name: str
    sampling_dist: str
    top_k_worst: int

    def __post_init__(self) -> None:
        self.manifests_dir = Path(self.manifests_dir)
        self.clips_dir = Path(self.clips_dir)
        self.output_dir = Path(self.output_dir)


def run_evaluation(config: VideoMAEEvalConfig):
    """Evaluate a fine-tuned VideoMAE model."""

    load_dotenv()

    dataset = load_dataset(
        "json",
        data_files={"test": str(config.manifests_dir / "test.jsonl")},
    )

    eval_dataset = dataset["test"]

    labels, label2id, id2label = load_target_label_mappings(config.manifests_dir)

    components = mlflow_transformers.load_model(
        config.model_uri, return_type="components"
    )
    model = components["model"].to(config.device)
    processor = components["image_processor"]

    compute_metrics = build_compute_metrics()

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_eval_batch_size=config.batch_size,
        dataloader_num_workers=config.num_workers,
        remove_unused_columns=False,
        report_to=["mlflow"],
        run_name=config.run_name,
    )

    data_collator = build_collate(
        processor,
        config.clips_dir,
        label2id,
        config.sample_frames,
        config.sampling_dist,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_input(from_huggingface(eval_dataset, name="test"), context="test")

        pred_logits, actual_ids, _ = trainer.predict(eval_dataset)

        pred_ids = compute_pred_ids(pred_logits)
        metrics = compute_metrics(pred_ids, actual_ids)

        mlflow.log_metrics(metrics)

        dummy_pred_ids = compute_dummy_pred_ids(actual_ids)
        dummy_metrics = compute_metrics(dummy_pred_ids, actual_ids)

        dummy_metrics_prefixed = {f"dummy_{k}": v for k, v in dummy_metrics.items()}

        mlflow.log_metrics(dummy_metrics_prefixed)

        dist_fig = get_target_distribution_figure(
            pred_ids, actual_ids, labels, id2label
        )

        mlflow.log_figure(dist_fig, "predicted_vs_actual_target_counts.png")

        cm_fig = get_confusion_matrix_figure(pred_ids, actual_ids, labels)
        mlflow.log_figure(cm_fig, "confusion_matrix.png")

        f1_fig = get_per_class_f1_figure(pred_ids, actual_ids, labels)
        mlflow.log_figure(f1_fig, "per_class_f1.png")

        worst_misses = get_top_k_worst_misses(
            pred_logits,
            actual_ids,
            eval_dataset.to_pandas(),
            id2label,
            top_k=config.top_k_worst,
        )

        mlflow.log_table(worst_misses, "worst_misses.json")
