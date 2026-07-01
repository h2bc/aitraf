"""Runtime settings for the AITRAF API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from dotenv import load_dotenv


CLASSIFICATION_MANIFEST = "manifests/trick_classification/test.jsonl"
AQA_MANIFEST = "manifests/score_prediction_ordinal/test.jsonl"
CLIPS_DIR = "data/clips"


@dataclass(frozen=True)
class TrickClassificationConfig:
    model_uri: str
    manifest_path: Path
    ground_truth_field: str = "trick"
    model_kind: str = "video_mae"


@dataclass(frozen=True)
class TrickAssessmentConfig:
    model_uri: str
    manifest_path: Path
    features_dir: Path
    frame_cache_dir: Path
    model_cache_dir: Path
    ground_truth_field: str = "execution_score"
    model_kind: str = "video_mae_temporal_fusion"


@dataclass(frozen=True)
class TrickAssessmentPreProcessingConfig:
    backbone: str
    num_clips: int
    sample_frames: int
    sampling_dist: str
    feature_cache_dir: Path
    id2label: dict[str, str]


@dataclass(frozen=True)
class DemoClipsConfig:
    download_enabled: bool = False
    force_download: bool = False


@dataclass(frozen=True)
class Settings:
    api_token: str | None
    device: str
    clips_dir: Path
    classification: TrickClassificationConfig
    aqa: TrickAssessmentConfig
    demo_clips: DemoClipsConfig = DemoClipsConfig()


def resolve_api_device(value: str) -> str:
    device = value.strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise RuntimeError("AITRAF_API_DEVICE must be one of: auto, cpu, cuda.")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("AITRAF_API_DEVICE=cuda but CUDA is not available.")
    return device


def resolve_env_flag(env: Mapping[str, str], name: str) -> bool:
    value = env.get(name)
    if value is None or not value.strip():
        return False

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be one of: 1, 0, true, false, yes, no, on, off.")


def load_settings(
    env: Mapping[str, str],
    *,
    root: Path,
) -> Settings:
    load_dotenv(root / ".env", override=False)

    device = resolve_api_device(env["AITRAF_API_DEVICE"])
    data_dir = Path(env["AITRAF_DATA_PATH"])
    storage_dir = Path(env["AITRAF_STORAGE_PATH"])

    features_dir = storage_dir / "data" / "video_mae_features"
    frame_cache_dir = storage_dir / "data" / "frame_cache"
    model_cache_dir = storage_dir / "models"

    return Settings(
        api_token=env["AITRAF_API_TOKEN"],
        device=device,
        clips_dir=storage_dir / CLIPS_DIR,
        classification=TrickClassificationConfig(
            model_uri=env["AITRAF_CLASSIFICATION_MODEL_URI"],
            manifest_path=data_dir / CLASSIFICATION_MANIFEST,
        ),
        aqa=TrickAssessmentConfig(
            model_uri=env["AITRAF_AQA_MODEL_URI"],
            manifest_path=data_dir / AQA_MANIFEST,
            features_dir=features_dir,
            frame_cache_dir=frame_cache_dir,
            model_cache_dir=model_cache_dir,
        ),
        demo_clips=DemoClipsConfig(
            download_enabled=resolve_env_flag(env, "AITRAF_API_DEMO_CLIPS_DOWNLOAD"),
            force_download=resolve_env_flag(env, "AITRAF_API_DEMO_CLIPS_FORCE_DOWNLOAD"),
        ),
    )


__all__ = [
    "DemoClipsConfig",
    "Settings",
    "TrickAssessmentConfig",
    "TrickAssessmentPreProcessingConfig",
    "TrickClassificationConfig",
    "load_settings",
    "resolve_api_device",
    "resolve_env_flag",
]
