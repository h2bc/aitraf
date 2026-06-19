"""Runtime settings for the AITRAF API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv


CLASSIFICATION_MANIFEST = "manifests/trick_classification/test.jsonl"
AQA_MANIFEST = "manifests/score_prediction_ordinal/test.jsonl"
CLIPS_DIR = "data/clips"


@dataclass(frozen=True)
class RegisteredModelReference:
    model_uri: str
    model_kind: str
    manifest_path: Path
    ground_truth_field: str


@dataclass(frozen=True)
class Settings:
    api_token: str | None
    mlflow_tracking_uri: str | None
    clips_dir: Path
    classification: RegisteredModelReference
    aqa: RegisteredModelReference


def load_settings(
    env: Mapping[str, str] | None = None,
    *,
    root: Path | None = None,
) -> Settings:
    repo_root = root or Path.cwd()
    load_dotenv(repo_root / ".env", override=False)
    source = env or os.environ

    data_dir = Path(source["AITRAF_DATA_PATH"])
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir

    storage_dir = Path(source["AITRAF_STORAGE_PATH"])
    if not storage_dir.is_absolute():
        storage_dir = repo_root / storage_dir

    return Settings(
        api_token=source.get("AITRAF_API_TOKEN"),
        mlflow_tracking_uri=source.get("MLFLOW_TRACKING_URI"),
        clips_dir=storage_dir / CLIPS_DIR,
        classification=RegisteredModelReference(
            model_uri=source["AITRAF_CLASSIFICATION_MODEL_URI"],
            model_kind="video_mae",
            manifest_path=data_dir / CLASSIFICATION_MANIFEST,
            ground_truth_field="trick",
        ),
        aqa=RegisteredModelReference(
            model_uri=source["AITRAF_AQA_MODEL_URI"],
            model_kind="ordinal",
            manifest_path=data_dir / AQA_MANIFEST,
            ground_truth_field="execution_score",
        ),
    )


__all__ = [
    "RegisteredModelReference",
    "Settings",
    "load_settings",
]
