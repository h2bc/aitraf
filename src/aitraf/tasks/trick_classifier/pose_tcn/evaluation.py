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
    EvalModel,
    EvalSet,
    calc_metrics_for_models,
    accuracy,
    f1_macro,
    compute_dummy_classification_pred_ids,
    compute_pred_ids,
    flatten_metrics_report,
    get_confusion_matrix_figure,
    get_per_class_f1_figure,
    get_target_distribution_figure,
    get_top_k_worst_misses,
    metrics_to_df,
)
from aitraf.tracking import build_training_params, params_to_df
from aitraf.models.pose_tcn import TCNClassifier
from aitraf.processing import load_target_label_mappings
from aitraf.processing.models.pose_tcn import process_sample
from aitraf.processing.utils import build_collate
from aitraf.tracking.models.pose_tcn import TRAINING_PARAM_MAP


@dataclass
class PoseTcnTrickClassificationEvalCfg:
    """Configuration for evaluating Pose TCN."""

    model_uri: str
    manifests_dir: Path | str
    vocab_path: Path | str
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
        config.vocab_path, "trick"
    )

    train_dataset = PoseTCNDataset(
        manifests_dir=config.manifests_dir,
        poses_dir=config.poses_dir,
        split="train",
    )
    test_dataset = PoseTCNDataset(
        manifests_dir=config.manifests_dir,
        poses_dir=config.poses_dir,
        split="test",
    )

    process_fn = partial(
        process_sample,
        num_frames=config.sample_frames,
        sampling_dist=config.sampling_dist,
        label_key="trick",
        label_transform=lambda label: label2id[str(label)],
    )

    collate_fn = build_collate(process_fn)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = mlflow.pytorch.load_model(config.model_uri)
    model = cast(TCNClassifier, model)
    model = model.to(config.device)
    model.eval()

    def predict_ids_and_labels(
        dataloader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return logits, pred_ids, label_ids

    _, train_pred_ids, train_label_ids = predict_ids_and_labels(train_dataloader)
    test_logits, test_pred_ids, test_label_ids = predict_ids_and_labels(test_dataloader)

    metrics_report = calc_metrics_for_models(
        eval_models=[
            EvalModel(
                name="dummy",
                sets=[
                    EvalSet(
                        name="train",
                        predictions=compute_dummy_classification_pred_ids(
                            train_label_ids
                        ),
                        labels=train_label_ids,
                    ),
                    EvalSet(
                        name="test",
                        predictions=compute_dummy_classification_pred_ids(
                            test_label_ids
                        ),
                        labels=test_label_ids,
                    ),
                ],
            ),
            EvalModel(
                name="pose_tcn",
                sets=[
                    EvalSet(
                        name="train",
                        predictions=train_pred_ids,
                        labels=train_label_ids,
                    ),
                    EvalSet(
                        name="test",
                        predictions=test_pred_ids,
                        labels=test_label_ids,
                    ),
                ],
            ),
        ],
        eval_metrics=(
            accuracy,
            f1_macro,
        ),
    )

    all_metrics = flatten_metrics_report(metrics_report)

    source_train_run_id = mlflow.models.get_model_info(config.model_uri).run_id
    source_train_params = build_training_params(
        source_train_run_id, TRAINING_PARAM_MAP
    ) | {"eval_sampling_dist": config.sampling_dist}

    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params(source_train_params)
        params_df = params_to_df(source_train_params)
        mlflow.log_table(params_df, "params_table.json")
        mlflow.log_input(
            from_pandas(pd.DataFrame(train_dataset.manifest_rows()), name="train"),
            context="train",
        )
        mlflow.log_input(
            from_pandas(pd.DataFrame(test_dataset.manifest_rows()), name="test"),
            context="test",
        )

        mlflow.log_metrics(all_metrics)
        metrics_df = metrics_to_df(metrics_report)
        mlflow.log_table(metrics_df, "metrics_table.json")

        dist_fig = get_target_distribution_figure(
            test_pred_ids,
            test_label_ids,
            label_names,
            id2label,
        )
        mlflow.log_figure(dist_fig, "predicted_vs_actual_target_counts.png")

        cm_fig = get_confusion_matrix_figure(test_pred_ids, test_label_ids, label_names)
        mlflow.log_figure(cm_fig, "confusion_matrix.png")

        f1_fig = get_per_class_f1_figure(test_pred_ids, test_label_ids, label_names)
        mlflow.log_figure(f1_fig, "per_class_f1.png")

        test_examples_df = pd.DataFrame(test_dataset.manifest_rows())
        all_misses = get_top_k_worst_misses(
            test_logits,
            test_label_ids,
            test_examples_df,
            id2label,
        )

        if not all_misses.empty:
            mlflow.log_table(all_misses, "all_misses.json")


__all__ = ["PoseTcnTrickClassificationEvalCfg", "run_evaluation"]
