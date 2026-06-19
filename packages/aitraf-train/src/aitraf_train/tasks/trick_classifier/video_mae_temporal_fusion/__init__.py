"""VideoMAE temporal-fusion trick-classifier task entrypoints."""

from .evaluation import VideoMaeTemporalFusionTrickClassificationEvalCfg, run_evaluation
from .training import VideoMaeTemporalFusionTrickClassificationTrainCfg, run_training

__all__ = [
    "VideoMaeTemporalFusionTrickClassificationEvalCfg",
    "VideoMaeTemporalFusionTrickClassificationTrainCfg",
    "run_evaluation",
    "run_training",
]
