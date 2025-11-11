"""Core package exposing shared project utilities."""

from . import dataset_schema
from .paths import (
    CHECKPOINTS_DIR,
    CONFIGS_DIR,
    DATA_DIR,
    LOGS_DIR,
    MANIFESTS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "MANIFESTS_DIR",
    "MODELS_DIR",
    "NOTEBOOKS_DIR",
    "RUNS_DIR",
    "CHECKPOINTS_DIR",
    "LOGS_DIR",
    "CONFIGS_DIR",
    "dataset_schema",
]
