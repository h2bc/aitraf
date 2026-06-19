"""Shared registered-model prediction helpers."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from fastapi import HTTPException

from aitraf_core.inference import predict_video_mae_label
from aitraf_core.utils import LoadedTransformersModel, load_transformers_model
from aitraf_api.config import RegisteredModelReference, Settings
from aitraf_api.schemas import DisplayResult, InferenceResult, ModelInfo, PredictionResult


LoadModel = Callable[[str, str | None], Any]
PredictVideo = Callable[..., tuple[str, float]]


def predict_manifest_row(
    *,
    ref: RegisteredModelReference,
    row: dict[str, Any],
    settings: Settings,
    load_model: LoadModel | None = None,
    predict_video: PredictVideo = predict_video_mae_label,
) -> InferenceResult:
    video_id = str(row["video_id"])
    loader = load_model or load_registered_model
    label, confidence = predict_video(
        loaded_model=loader(ref.model_uri, settings.mlflow_tracking_uri),
        video_id=video_id,
        local_clips_dir=settings.clips_dir,
    )
    return InferenceResult(
        video_id=video_id,
        prediction=PredictionResult(label=label, confidence=confidence),
        ground_truth=DisplayResult(label=str(row[ref.ground_truth_field])),
        model=ModelInfo(kind=ref.model_kind),
    )


@lru_cache(maxsize=4)
def load_registered_model(
    model_uri: str,
    tracking_uri: str | None,
) -> LoadedTransformersModel:
    if not tracking_uri:
        raise HTTPException(
            status_code=503,
            detail="MLFLOW_TRACKING_URI is not configured",
        )

    try:
        return load_transformers_model(
            model_uri,
            tracking_uri=tracking_uri,
        )
    except Exception as exc:  # noqa: BLE001 - wrap MLflow/registry failures
        raise HTTPException(
            status_code=503,
            detail=f"Registered transformers model is unavailable: {model_uri}",
        ) from exc


def validate_clip_exists(settings: Settings, video_id: str) -> None:
    clip_path = settings.clips_dir / video_id
    if not clip_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Video file is unavailable: {clip_path}",
        )


__all__ = [
    "LoadModel",
    "PredictVideo",
    "load_registered_model",
    "predict_manifest_row",
    "validate_clip_exists",
]
