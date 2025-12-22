"""Pose TCN trick-classifier task entrypoints."""

from .training import PoseTcnTrickClassificationCfg, run_training
from .evaluation import PoseTCNEvalConfig, run_evaluation

__all__ = [
    "PoseTcnTrickClassificationCfg",
    "PoseTCNEvalConfig",
    "run_training",
    "run_evaluation",
]
