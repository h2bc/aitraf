"""Pose TCN trick-classifier task entrypoints."""

from .training import PoseTcnTrickClassificationTrainCfg, run_training
from .evaluation import PoseTcnTrickClassificationEvalCfg, run_evaluation

__all__ = [
    "PoseTcnTrickClassificationTrainCfg",
    "PoseTcnTrickClassificationEvalCfg",
    "run_training",
    "run_evaluation",
]
