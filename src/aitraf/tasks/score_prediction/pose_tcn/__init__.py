"""Pose TCN score-prediction task entrypoints."""

from .training import PoseTcnScorePredictionCfg, run_training

__all__ = [
    "PoseTcnScorePredictionCfg",
    "run_training",
]
