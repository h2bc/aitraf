"""Pose TCN ordinal score-prediction task entrypoints."""

from .evaluation import PoseTcnScorePredictionOrdinalEvalCfg, run_evaluation
from .training import PoseTcnScorePredictionOrdinalTrainCfg, run_training

__all__ = [
    "PoseTcnScorePredictionOrdinalEvalCfg",
    "PoseTcnScorePredictionOrdinalTrainCfg",
    "run_evaluation",
    "run_training",
]
