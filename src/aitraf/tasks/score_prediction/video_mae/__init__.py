"""VideoMAE score-prediction task entrypoints."""

from .evaluation import VideoMaeScorePredictionEvalCfg, run_evaluation
from .training import VideoMaeScorePredictionTrainCfg, run_training

__all__ = [
    "VideoMaeScorePredictionTrainCfg",
    "VideoMaeScorePredictionEvalCfg",
    "run_training",
    "run_evaluation",
]
