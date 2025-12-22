"""Pose TCN training loop with Lightning + MLflow logging."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.pytorch
from mlflow.data import from_pandas
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, Subset

from aitraf.datasets.pose_tcn import PoseTCNDataset
from aitraf.models.pose_tcn import TCNClassifier
from aitraf.processing import load_target_label_mappings
from aitraf.processing.models.pose_tcn import process_sample
from aitraf.processing.utils import build_collate


@dataclass
class PoseTcnTrickClassificationCfg:
    """Configuration for Pose TCN training."""

    task_name: str
    model_name: str
    manifests_dir: Path | str
    vocab_path: Path | str
    target_col: str
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
    max_train_samples: int | None = None

    def __post_init__(self) -> None:
        self.manifests_dir = Path(self.manifests_dir)
        self.poses_dir = Path(self.poses_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def run_training(config: PoseTcnTrickClassificationCfg) -> str:
    """Train the Pose TCN classifier and log artifacts to MLflow."""
    labels, label2id, _ = load_target_label_mappings(
        config.vocab_path, config.target_col
    )

    process_fn = partial(
        process_sample,
        num_frames=config.sample_frames,
        sampling_dist=config.sampling_dist,
        label_transform=lambda label: label2id[str(label)],
    )

    collate_fn = build_collate(process_fn)

    pin_memory = config.accelerator != "cpu"

    train_dataset = PoseTCNDataset(
        manifests_dir=config.manifests_dir,
        poses_dir=config.poses_dir,
        target_column=config.target_col,
        split="train",
    )

    if config.max_train_samples is not None:
        max_count = min(config.max_train_samples, len(train_dataset))
        train_dataset = Subset(train_dataset, range(max_count))

    val_dataset = PoseTCNDataset(
        manifests_dir=config.manifests_dir,
        poses_dir=config.poses_dir,
        target_column=config.target_col,
        split="val",
    )

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

    model = TCNClassifier(
        feature_dim=feature_dim,
        num_classes=num_classes,
        learning_rate=config.learning_rate,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(config.output_dir),
        filename="pose-tcn-{epoch:02d}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    early_stop = EarlyStopping(
        monitor="val_acc",
        mode="max",
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

        mlflow.log_input(
            from_pandas(
                pd.DataFrame(train_loader.dataset.manifest_rows()), name="train"
            ),
            context="train",
        )
        mlflow.log_input(
            from_pandas(
                pd.DataFrame(val_loader.dataset.manifest_rows()), name="validation"
            ),
            context="validation",
        )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_checkpoint = checkpoint_callback.best_model_path
        if not best_checkpoint:
            raise RuntimeError(
                "Checkpoint was not saved; unable to log model to MLflow."
            )

        mlflow.log_artifact(best_checkpoint, artifact_path="checkpoints")

        exported_model = (
            TCNClassifier.load_from_checkpoint(best_checkpoint).cpu().eval()
        )

        sample_input = first_batch["inputs"][:1].cpu().numpy().astype("float32")

        model_info = mlflow.pytorch.log_model(
            exported_model,
            name=f"{config.task_name}_{config.model_name}",
            input_example=sample_input,
        )

        return model_info.model_uri
