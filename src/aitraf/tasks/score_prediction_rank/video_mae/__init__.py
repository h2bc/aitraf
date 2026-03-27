"""VideoMAE score-prediction-rank task entrypoints."""

from .evaluation import VideoMaeScorePredictionRankEvalCfg, run_evaluation
from .training import VideoMaeScorePredictionRankTrainCfg, run_training

__all__ = [
    "VideoMaeScorePredictionRankEvalCfg",
    "VideoMaeScorePredictionRankTrainCfg",
    "run_evaluation",
    "run_training",
]
