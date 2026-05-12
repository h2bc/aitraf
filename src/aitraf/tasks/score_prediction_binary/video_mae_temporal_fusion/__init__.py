"""VideoMAE temporal-fusion binary score-prediction task entrypoints."""

from .evaluation import VideoMaeTemporalFusionScorePredictionBinaryEvalCfg, run_evaluation
from .training import VideoMaeTemporalFusionScorePredictionBinaryTrainCfg, run_training

__all__ = [
    "VideoMaeTemporalFusionScorePredictionBinaryEvalCfg",
    "VideoMaeTemporalFusionScorePredictionBinaryTrainCfg",
    "run_evaluation",
    "run_training",
]
