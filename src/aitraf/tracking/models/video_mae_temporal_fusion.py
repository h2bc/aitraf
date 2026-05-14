"""VideoMAE temporal-fusion tracking parameter map."""

TRAINING_PARAM_MAP = {
    "num_workers": "num_workers",
    "backbone": "backbone",
    "num_frames": "num_frames",
    "metric_for_best_model": "metric_for_best_model",
    "max_epochs": "max_epochs",
    "learning_rate": "learning_rate",
    "batch_size": "batch_size",
    "train_sampling_dist": "train_sampling_dist",
    "frozen": "frozen",
    "num_clips": "num_clips",
    "fusion_layers": "fusion_layers",
    "fusion_heads": "fusion_heads",
    "fusion_dropout": "fusion_dropout",
}

__all__ = ["TRAINING_PARAM_MAP"]
