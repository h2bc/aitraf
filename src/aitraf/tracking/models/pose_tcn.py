"""Pose TCN tracking parameter map."""

TRAINING_PARAM_MAP = {
    "num_workers": "num_workers",
    "num_frames": "sample_frames",
    "metric_for_best_model": "metric_for_best_model",
    "max_epochs": "max_epochs",
    "batch_size": "batch_size",
    "sampling_dist": "sampling_dist",
    "feature_dim": "feature_dim",
    "hidden_dim": "hidden_dim",
    "num_layers": "num_layers",
    "kernel_size": "kernel_size",
    "dropout": "dropout",
}

__all__ = ["TRAINING_PARAM_MAP"]
