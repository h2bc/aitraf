"""VideoMAE trick-classifier task entrypoints."""

from .training import VideoMaeTrickClassificationCfg, run_training
from .evaluation import VideoMAEEvalConfig, run_evaluation

__all__ = [
    "VideoMaeTrickClassificationCfg",
    "VideoMAEEvalConfig",
    "run_training",
    "run_evaluation",
]
