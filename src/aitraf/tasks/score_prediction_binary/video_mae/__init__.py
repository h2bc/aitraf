"""VideoMAE binary score-prediction task entrypoints."""

from .evaluation import VideoMaeScorePredictionBinaryEvalCfg, run_evaluation
from .training import VideoMaeScorePredictionBinaryTrainCfg, run_training

__all__ = [
    "VideoMaeScorePredictionBinaryTrainCfg",
    "VideoMaeScorePredictionBinaryEvalCfg",
    "run_training",
    "run_evaluation",
]
