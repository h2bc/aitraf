"""Pose TCN tracking parameter map."""

TRAINING_PARAM_MAP = {
    "num_workers": "num_workers",
    "num_frames": "sample_frames",
    "metric_for_best_model": "metric_for_best_model",
    "max_epochs": "max_epochs",
    "learning_rate": "learning_rate",
    "batch_size": "batch_size",
    "train_sampling_dist": "train_sampling_dist",
    "feature_dim": "feature_dim",
    "hidden_dim": "hidden_dim",
    "num_layers": "num_layers",
    "kernel_size": "kernel_size",
    "dropout": "dropout",
    "loss": "loss",
    "use_class_weights": "use_class_weights",
    "best_model_metric": "best_model_metric",
    "seed": "seed",
}

__all__ = ["TRAINING_PARAM_MAP"]
