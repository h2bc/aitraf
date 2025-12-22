"""Pose TCN score-prediction task entrypoints."""

from .training import PoseTcnScorePredictionCfg, run_training
from .evaluation import PoseTcnScorePredictionEvalCfg, run_evaluation

__all__ = [
    "PoseTcnScorePredictionCfg",
    "PoseTcnScorePredictionEvalCfg",
    "run_training",
    "run_evaluation",
]
