"""VideoMAE score-prediction-rank task entrypoints."""

from .training import VideoMaeScorePredictionRankTrainCfg, run_training

__all__ = [
    "VideoMaeScorePredictionRankTrainCfg",
    "run_training",
]
