"""VideoMAE score prediction pairwise task entrypoints."""

from .evaluation import VideoMaeScorePredictionPairwiseEvalCfg, run_evaluation
from .training import VideoMaeScorePredictionPairwiseTrainCfg, run_training

__all__ = [
    "VideoMaeScorePredictionPairwiseEvalCfg",
    "VideoMaeScorePredictionPairwiseTrainCfg",
    "run_evaluation",
    "run_training",
]
