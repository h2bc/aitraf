"""Pose TCN score-prediction task entrypoints."""

from .training import PoseTCNRegressionTrainingConfig, run_training

__all__ = [
    "PoseTCNRegressionTrainingConfig",
    "run_training",
]
