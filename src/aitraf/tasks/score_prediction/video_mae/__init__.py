"""VideoMAE score-prediction task entrypoints."""

from .training import VideoMaeScorePredictionTrainCfg, run_training

__all__ = ["VideoMaeScorePredictionTrainCfg", "run_training"]
