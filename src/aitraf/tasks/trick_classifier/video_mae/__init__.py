"""VideoMAE trick-classifier task entrypoints."""

from .training import VideoMAETrainingConfig, run_training
from .evaluation import VideoMAEEvalConfig, run_evaluation

__all__ = [
    "VideoMAETrainingConfig",
    "VideoMAEEvalConfig",
    "run_training",
    "run_evaluation",
]
