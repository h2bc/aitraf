"""Pose TCN evaluation pipeline for score prediction."""

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
    build_regression_metrics,
    compute_dummy_regression_preds,
    get_predicted_vs_actual_scatter_figure,
)
from aitraf.models.pose_tcn import TCNRegressor
from aitraf.processing.models.pose_tcn import process_sample
from aitraf.processing.utils import build_collate


@dataclass
class PoseTcnScorePredictionEvalCfg:
    """Configuration for evaluating Pose TCN score prediction."""

    model_uri: str
    manifests_dir: Path | str
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
        self.poses_dir = Path(self.poses_dir)


def run_evaluation(config: PoseTcnScorePredictionEvalCfg) -> None:

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
    model = cast(TCNRegressor, model)
    model = model.to(config.device)
    model.eval()

    compute_metrics = build_regression_metrics()

    preds_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(config.device)
            labels = batch["labels"].to(config.device).float()
            preds = model(inputs)
            preds_list.append(preds.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    predictions = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    metrics = compute_metrics(predictions, labels)
    dummy_metrics = compute_metrics(compute_dummy_regression_preds(labels), labels)
    dummy_metrics = {f"dummy_{k}": v for k, v in dummy_metrics.items()}

    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_input(
            from_pandas(pd.DataFrame(dataset.manifest_rows()), name="test"),
            context="test",
        )

        mlflow.log_metrics(metrics)
        mlflow.log_metrics(dummy_metrics)
        scatter_fig = get_predicted_vs_actual_scatter_figure(predictions, labels)
        mlflow.log_figure(scatter_fig, "predicted_vs_actual.png")


__all__ = ["PoseTcnScorePredictionEvalCfg", "run_evaluation"]
