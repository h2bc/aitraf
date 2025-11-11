from pathlib import Path
import os

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"
RUNS_DIR = PROJECT_ROOT / "runs"
MANIFESTS_DIR = DATA_DIR / "manifests"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
CHECKPOINTS_DIR = RUNS_DIR / "checkpoints"
LOGS_DIR = RUNS_DIR / "logs"


def export_env() -> None:
    os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    os.environ.setdefault("DATA_DIR", str(DATA_DIR))
    os.environ.setdefault("MODELS_DIR", str(MODELS_DIR))
    os.environ.setdefault("RUNS_DIR", str(RUNS_DIR))
    os.environ.setdefault("CONFIGS_DIR", str(CONFIGS_DIR))


export_env()


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "MANIFESTS_DIR",
    "NOTEBOOKS_DIR",
    "MODELS_DIR",
    "RUNS_DIR",
    "CHECKPOINTS_DIR",
    "LOGS_DIR",
    "CONFIGS_DIR",
    "export_env",
]
