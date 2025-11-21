"""VideoMAE evaluation pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv
from mlflow.data import from_huggingface
import mlflow
from mlflow import transformers as mlflow_transformers
from transformers import (
    TrainingArguments,
    Trainer,
)

from aitraf.video_mae.common import build_collate, load_target_label_mappings
from aitraf.video_mae.metrics import build_compute_metrics


@dataclass
class VideoMAEEvalConfig:
    """Configuration for VideoMAE evaluation."""

    backbone: str
    model_id: str
    manifests_dir: Path | str
    clips_dir: Path | str
    batch_size: int
    num_workers: int
    sample_frames: int
    device: str
    output_dir: Path | str
    run_name: str
    experiment_name: str

    def __post_init__(self) -> None:
        self.manifests_dir = Path(self.manifests_dir)
        self.clips_dir = Path(self.clips_dir)
        self.output_dir = Path(self.output_dir)


def run_evaluation(config: VideoMAEEvalConfig) -> tuple[str, dict[str, Any]]:
    """Evaluate a fine-tuned VideoMAE model."""

    load_dotenv()

    dataset = load_dataset(
        "json",
        data_files={"test": str(config.manifests_dir / "test.jsonl")},
    )

    eval_dataset = dataset["test"]

    label2id, _ = load_target_label_mappings(config.manifests_dir)

    components = mlflow_transformers.load_model(
        f"models:/{config.model_id}", return_type="components"
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
        processor, config.clips_dir, label2id, config.sample_frames
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name) as run:
        mlflow.log_input(from_huggingface(eval_dataset, name="test"), context="test")
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)

        return run.info.run_id, metrics
