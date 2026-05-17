"""VideoMAE tracking parameter map."""

TRAINING_PARAM_MAP = {
    "num_workers": "dataloader_num_workers",
    "backbone": "_name_or_path",
    "num_frames": "num_frames",
    "metric_for_best_model": "metric_for_best_model",
    "max_epochs": "num_train_epochs",
    "batch_size": "per_device_train_batch_size",
    "image_size": "image_size",
    "train_sampling_dist": "train_sampling_dist",
    "frozen": "frozen",
    "loss": "loss",
    "use_class_weights": "use_class_weights",
}

__all__ = ["TRAINING_PARAM_MAP"]
