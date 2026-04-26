"""MLflow parameter helpers for evaluation runs."""

from __future__ import annotations

from mlflow.tracking import MlflowClient


def build_training_params(
    run_id: str,
) -> dict[str, str]:
    """Return the selected evaluation params from a training run."""

    training_run = MlflowClient().get_run(run_id)
    params = training_run.data.params

    return {
        "num_workers": params["dataloader_num_workers"],
        "num_frames": params["num_frames"],
        "metric_for_best_model": params["metric_for_best_model"],
        "trained_epochs": params["epoch"],
        "max_epochs": params["num_train_epochs"],
        "batch_size": params["per_device_train_batch_size"],
        "image_size": params["image_size"],
    }


__all__ = ["build_training_params"]
