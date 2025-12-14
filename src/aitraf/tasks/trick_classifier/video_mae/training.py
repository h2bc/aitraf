"""VideoMAE finte-tuning pipeline"""

from dataclasses import dataclass
from pathlib import Path


from transformers import (
    AutoConfig,
    AutoModelForVideoClassification,
    EvalPrediction,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    VideoMAEImageProcessor,
)

from aitraf.processing.video_mae import build_collate, process_clip
from aitraf.processing import load_target_label_mappings
from aitraf.metrics import build_compute_metrics, compute_pred_ids

from datasets import load_dataset

from dotenv import load_dotenv
import mlflow
from mlflow.data import from_huggingface


@dataclass
class VideoMAETrainingConfig:
    """Minimal configuration for the current skeleton run."""

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


def run_training(config: VideoMAETrainingConfig) -> str:
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

    labels, label2id, id2label = load_target_label_mappings(config.manifests_dir)

    processor = VideoMAEImageProcessor.from_pretrained(
        config.backbone, cache_dir=str(config.model_cache_dir)
    )
    model_config = AutoConfig.from_pretrained(
        config.backbone,
        cache_dir=str(config.model_cache_dir),
        trust_remote_code=True,
        label2id=label2id,
        id2label=id2label,
        num_labels=len(labels),
    )

    model = AutoModelForVideoClassification.from_pretrained(
        config.backbone,
        config=model_config,
        cache_dir=str(config.model_cache_dir),
        trust_remote_code=True,
    ).to(config.device)

    if config.freeze_backbone:
        for param in model.base_model.parameters():
            param.requires_grad = False

    compute_metrics = build_compute_metrics()

    def trainer_compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
        pred_logits, actual_ids = prediction

        pred_ids = compute_pred_ids(pred_logits)

        return compute_metrics(pred_ids, actual_ids)

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
        metric_for_best_model="accuracy",
        greater_is_better=True,
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
        mlflow.log_input(
            from_huggingface(dataset["train"], name="train"), context="training"
        )
        mlflow.log_input(
            from_huggingface(dataset["validation"], name="validation"),
            context="validation",
        )

        trainer.train()

        sample_clip = process_clip(
            dataset["train"][0],
            processor,
            config.clips_dir,
            label2id,
            config.sample_frames,
            config.sampling_dist,
        )

        model_info = mlflow.transformers.log_model(
            transformers_model={
                "model": model,
                "image_processor": processor,
            },
            name=f"{config.experiment_name}_{config.run_name}",
            input_example={
                "pixel_values": sample_clip["pixel_values"]
                .unsqueeze(0)
                .numpy()
                .tolist()
            },
        )

        return model_info.model_uri
