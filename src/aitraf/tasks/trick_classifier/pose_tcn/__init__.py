"""Pose TCN trick-classifier task entrypoints."""

from .training import PoseTCNTrainingConfig, run_training
from .evaluation import PoseTCNEvalConfig, run_evaluation

__all__ = [
    "PoseTCNTrainingConfig",
    "PoseTCNEvalConfig",
    "run_training",
    "run_evaluation",
]
