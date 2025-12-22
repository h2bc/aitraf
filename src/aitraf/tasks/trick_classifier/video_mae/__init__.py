"""VideoMAE trick-classifier task entrypoints."""

from .training import VideoMaeTrickClassificationTrainCfg, run_training
from .evaluation import VideoMaeTrickClassificationEvalCfg, run_evaluation

__all__ = [
    "VideoMaeTrickClassificationTrainCfg",
    "VideoMaeTrickClassificationEvalCfg",
    "run_training",
    "run_evaluation",
]
