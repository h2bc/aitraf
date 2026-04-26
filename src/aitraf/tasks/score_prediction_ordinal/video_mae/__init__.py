"""VideoMAE ordinal score-prediction task entrypoints."""

from .evaluation import VideoMaeScorePredictionOrdinalEvalCfg, run_evaluation
from .training import VideoMaeScorePredictionOrdinalTrainCfg, run_training

__all__ = [
    "VideoMaeScorePredictionOrdinalEvalCfg",
    "VideoMaeScorePredictionOrdinalTrainCfg",
    "run_evaluation",
    "run_training",
]
