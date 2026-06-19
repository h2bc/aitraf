"""Shared FastAPI dependency helpers."""

from __future__ import annotations

from fastapi import Request

from aitraf_core.inference import predict_video_mae_label
from aitraf_api.config import Settings
from aitraf_api.prediction import LoadModel, PredictVideo


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_load_model(request: Request) -> LoadModel | None:
    return request.app.state.load_model


def get_predict_video(request: Request) -> PredictVideo:
    return request.app.state.predict_video or predict_video_mae_label


__all__ = [
    "get_load_model",
    "get_predict_video",
    "get_settings",
]
