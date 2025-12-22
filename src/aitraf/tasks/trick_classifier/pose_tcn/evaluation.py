"""Pose TCN evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import cast

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from mlflow.data import from_pandas
from torch.utils.data import DataLoader

from aitraf.datasets.pose_tcn import PoseTCNDataset
from aitraf.metrics import (
    build_classification_metrics,
    compute_pred_ids,
    compute_dummy_classification_pred_ids,
    get_confusion_matrix_figure,
    get_per_class_f1_figure,
    get_target_distribution_figure,
    get_top_k_worst_misses,
)
from aitraf.models.pose_tcn import TCNClassifier
from aitraf.processing import load_target_label_mappings
from aitraf.processing.models.pose_tcn import process_sample
from aitraf.processing.utils import build_collate


@dataclass
class PoseTcnTrickClassificationEvalCfg:
    """Configuration for evaluating Pose TCN."""

    model_uri: str
    manifests_dir: Path | str
    vocab_path: Path | str
    target_col: str
    poses_dir: Path | str
    batch_size: int
    num_workers: int
    sample_frames: int
    sampling_dist: str
    device: str
    experiment_name: str
    run_name: str
    top_k_worst: int

    def __post_init__(self) -> None:
        self.manifests_dir = Path(self.manifests_dir)
        self.vocab_path = Path(self.vocab_path)
        self.poses_dir = Path(self.poses_dir)


def run_evaluation(config: PoseTcnTrickClassificationEvalCfg) -> None:
    label_names, label2id, id2label = load_target_label_mappings(
        config.vocab_path, config.target_col
    )

    dataset = PoseTCNDataset(
        manifests_dir=config.manifests_dir,
        poses_dir=config.poses_dir,
        target_column=config.target_col,
        split="test",
    )

    process_fn = partial(
        process_sample,
        num_frames=config.sample_frames,
        sampling_dist=config.sampling_dist,
        label_transform=lambda label: label2id[str(label)],
    )

    collate_fn = build_collate(process_fn)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = mlflow.pytorch.load_model(config.model_uri)
    model = cast(TCNClassifier, model)
    model = model.to(config.device)
    model.eval()

    compute_metrics = build_classification_metrics()

    logits_list: list[np.ndarray] = []
    label_ids_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(config.device)
            batch_labels = batch["labels"].to(config.device)
            batch_logits = model(inputs)
            logits_list.append(batch_logits.cpu().numpy())
            label_ids_list.append(batch_labels.cpu().numpy())

    logits = np.concatenate(logits_list, axis=0)
    label_ids = np.concatenate(label_ids_list, axis=0)
    pred_ids = compute_pred_ids(logits)

    metrics = compute_metrics(pred_ids, label_ids)
    dummy_metrics = compute_metrics(
        compute_dummy_classification_pred_ids(label_ids), label_ids
    )
    dummy_metrics = {f"dummy_{k}": v for k, v in dummy_metrics.items()}

    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_input(
            from_pandas(pd.DataFrame(dataset.manifest_rows()), name="test"),
            context="test",
        )

        mlflow.log_metrics(metrics)
        mlflow.log_metrics(dummy_metrics)

        dist_fig = get_target_distribution_figure(
            pred_ids, label_ids, label_names, id2label
        )
        mlflow.log_figure(dist_fig, "predicted_vs_actual_target_counts.png")

        cm_fig = get_confusion_matrix_figure(pred_ids, label_ids, label_names)
        mlflow.log_figure(cm_fig, "confusion_matrix.png")

        f1_fig = get_per_class_f1_figure(pred_ids, label_ids, label_names)
        mlflow.log_figure(f1_fig, "per_class_f1.png")

        worst_misses = get_top_k_worst_misses(
            logits,
            label_ids,
            pd.DataFrame(dataset.manifest_rows()),
            id2label,
            top_k=config.top_k_worst,
        )

        if not worst_misses.empty:
            mlflow.log_table(worst_misses, "worst_misses.json")


__all__ = ["PoseTcnTrickClassificationEvalCfg", "run_evaluation"]
