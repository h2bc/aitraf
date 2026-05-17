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
    "fusion_queries": "fusion_queries",
    "query_init_std": "query_init_std",
    "fusion_dropout": "fusion_dropout",
    "ordinal_loss": "ordinal_loss",
    "use_class_weights": "use_class_weights",
    "best_model_metric": "best_model_metric",
}

__all__ = ["TRAINING_PARAM_MAP"]
