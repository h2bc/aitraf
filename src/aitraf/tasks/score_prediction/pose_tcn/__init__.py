"""Pose TCN score-prediction task entrypoints."""

from .training import PoseTcnScorePredictionTrainCfg, run_training
from .evaluation import PoseTcnScorePredictionEvalCfg, run_evaluation

__all__ = [
    "PoseTcnScorePredictionTrainCfg",
    "PoseTcnScorePredictionEvalCfg",
    "run_training",
    "run_evaluation",
]
