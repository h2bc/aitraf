"""Pose TCN training loop for ordinal score prediction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
from dlordinal.losses import CDWCELoss
from dotenv import load_dotenv
from mlflow.data import from_pandas
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch import nn
from torch.utils.data import DataLoader

from aitraf_train.datasets.pose_tcn import PoseTCNDataset, PoseTCNSubset
from aitraf_train.metrics import calc_metrics
from aitraf_train.models.pose_tcn import TCNClassifier
from aitraf_core.processing import (
    build_class_weights,
    build_label_transform,
    load_target_label_mappings,
)
from aitraf_core.processing.models.pose_tcn import process_sample
from aitraf_core.processing.utils import build_collate
from aitraf_train.tasks.score_prediction_ordinal.metrics import amae, mae, qwk


@dataclass
class PoseTcnScorePredictionOrdinalTrainCfg:
    """Configuration for Pose TCN ordinal score prediction training."""

    task_name: str
    model_name: str
    manifests_dir: Path | str
    vocab_path: Path | str
    poses_dir: Path | str
    batch_size: int
    num_workers: int
    sample_frames: int
    sampling_dist: str
    learning_rate: float
    hidden_dim: int
    num_layers: int
    kernel_size: int
    dropout: float
    max_epochs: int
    accelerator: str
    early_stopping_patience: int
    experiment_name: str
    run_name: str
    output_dir: Path | str
    loss: str
    use_class_weights: bool
    best_model_metric: str
    seed: int
    max_train_samples: int | None = None

    def __post_init__(self) -> None:
        self.manifests_dir = Path(self.manifests_dir)
        self.vocab_path = Path(self.vocab_path)
        self.poses_dir = Path(self.poses_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def run_training(config: PoseTcnScorePredictionOrdinalTrainCfg) -> str:
    """Train the Pose TCN ordinal classifier and log artifacts to MLflow."""

    load_dotenv()
    seed_everything(config.seed, workers=True)

    labels, label2id, _ = load_target_label_mappings(
        config.vocab_path, "execution_score"
    )

    label_transform = build_label_transform(label2id)
    process_fn = partial(
        process_sample,
        num_frames=config.sample_frames,
        sampling_dist=config.sampling_dist,
        label_key="execution_score",
        label_transform=label_transform,
    )
    collate_fn = build_collate(process_fn)

    train_dataset: PoseTCNDataset | PoseTCNSubset = PoseTCNDataset(
        manifests_dir=config.manifests_dir,
        poses_dir=config.poses_dir,
        split="train",
    )
    if config.max_train_samples is not None:
        max_count = min(config.max_train_samples, len(train_dataset))
        train_dataset = PoseTCNSubset(train_dataset, range(max_count))

    val_dataset = PoseTCNDataset(
        manifests_dir=config.manifests_dir,
        poses_dir=config.poses_dir,
        split="val",
    )

    pin_memory = config.accelerator != "cpu"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    first_batch = next(iter(train_loader))
    feature_dim = first_batch["inputs"].shape[-1]
    num_classes = len(labels)
    class_weights = (
        build_class_weights(
            [
                label_transform(row["execution_score"])
                for row in train_dataset.manifest_rows()
            ],
            num_labels=num_classes,
            device="cpu",
        )
        if config.use_class_weights
        else None
    )

    metrics_fn = _build_metrics_fn()
    loss_fn = _build_loss(
        loss_name=config.loss,
        num_labels=num_classes,
        class_weights=class_weights,
    )
    model = TCNClassifier(
        feature_dim=feature_dim,
        num_classes=num_classes,
        learning_rate=config.learning_rate,
        metrics_fn=metrics_fn,
        loss_fn=loss_fn,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
    )

    metric_for_best_model, mode = _best_model_metric(config.best_model_metric)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(config.output_dir),
        filename="pose-tcn-ordinal-{epoch:02d}-{" + metric_for_best_model + ":.3f}",
        monitor=metric_for_best_model,
        mode=mode,
        save_top_k=1,
    )
    early_stop = EarlyStopping(
        monitor=metric_for_best_model,
        mode=mode,
        patience=config.early_stopping_patience,
        verbose=True,
    )

    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name=config.run_name) as run:
        mlflow_logger = MLFlowLogger(
            experiment_name=config.experiment_name,
            run_id=run.info.run_id,
        )
        trainer = Trainer(
            accelerator=config.accelerator,
            max_epochs=config.max_epochs,
            logger=mlflow_logger,
            default_root_dir=str(config.output_dir),
            callbacks=[checkpoint_callback, early_stop],
            enable_progress_bar=True,
        )

        mlflow.log_params(_training_params(config, feature_dim, metric_for_best_model))
        mlflow.log_input(
            from_pandas(pd.DataFrame(train_dataset.manifest_rows()), name="train"),
            context="train",
        )
        mlflow.log_input(
            from_pandas(pd.DataFrame(val_dataset.manifest_rows()), name="validation"),
            context="validation",
        )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_checkpoint = checkpoint_callback.best_model_path
        if not best_checkpoint:
            raise RuntimeError(
                "Checkpoint was not saved; unable to log model to MLflow."
            )

        mlflow.log_artifact(best_checkpoint, artifact_path="checkpoints")

        export_loss_fn = _build_loss(
            loss_name=config.loss,
            num_labels=num_classes,
            class_weights=class_weights,
        )
        exported_model = (
            TCNClassifier.load_from_checkpoint(
                best_checkpoint,
                metrics_fn=metrics_fn,
                loss_fn=export_loss_fn,
            )
            .cpu()
            .eval()
        )
        sample_input = first_batch["inputs"][:1].cpu().numpy().astype("float32")
        model_info = mlflow.pytorch.log_model(
            exported_model,
            name=f"{config.task_name}_{config.model_name}",
            input_example=sample_input,
        )
        return model_info.model_uri


def _build_metrics_fn():
    return lambda predictions, labels: calc_metrics(
        predictions,
        labels,
        (
            amae,
            mae,
            qwk,
        ),
    )


def _training_params(
    config: PoseTcnScorePredictionOrdinalTrainCfg,
    feature_dim: int,
    metric_for_best_model: str,
) -> dict[str, object]:
    return {
        "num_workers": config.num_workers,
        "num_frames": config.sample_frames,
        "metric_for_best_model": metric_for_best_model,
        "max_epochs": config.max_epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "train_sampling_dist": config.sampling_dist,
        "feature_dim": feature_dim,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "kernel_size": config.kernel_size,
        "dropout": config.dropout,
        "loss": config.loss,
        "use_class_weights": config.use_class_weights,
        "best_model_metric": config.best_model_metric,
        "seed": config.seed,
    }


def _build_loss(
    *,
    loss_name: str,
    num_labels: int,
    class_weights,
) -> nn.Module:
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)

    if loss_name in {"cdwce", "cwde"}:
        return CDWCELoss(num_classes=num_labels, alpha=0.5, weight=class_weights)

    raise ValueError(
        f"Unsupported loss '{loss_name}'. Expected 'cross_entropy' or 'cdwce'."
    )


def _best_model_metric(metric_name: str) -> tuple[str, str]:
    metric = metric_name.strip().lower()
    supported_metrics = {
        "loss": ("val_loss", "min"),
        "amae": ("val_amae", "min"),
        "mae": ("val_mae", "min"),
        "qwk": ("val_qwk", "max"),
    }

    if metric not in supported_metrics:
        raise ValueError(
            f"Unsupported best_model_metric '{metric_name}'. "
            f"Expected one of: {sorted(supported_metrics)}."
        )
    return supported_metrics[metric]


__all__ = ["PoseTcnScorePredictionOrdinalTrainCfg", "run_training"]
