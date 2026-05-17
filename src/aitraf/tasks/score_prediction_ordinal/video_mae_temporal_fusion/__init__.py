"""VideoMAE temporal-fusion ordinal score-prediction task entrypoints."""

from .evaluation import (
    VideoMaeTemporalFusionScorePredictionOrdinalEvalCfg,
    run_evaluation,
)
from .training import VideoMaeTemporalFusionScorePredictionOrdinalTrainCfg, run_training

__all__ = [
    "VideoMaeTemporalFusionScorePredictionOrdinalEvalCfg",
    "VideoMaeTemporalFusionScorePredictionOrdinalTrainCfg",
    "run_evaluation",
    "run_training",
]
