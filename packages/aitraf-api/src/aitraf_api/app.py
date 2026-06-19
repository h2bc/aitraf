"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from aitraf_api.config import Settings, load_settings
from aitraf_api.features import router
from aitraf_api.prediction import LoadModel, PredictVideo


def create_app(
    *,
    settings: Settings | None = None,
    load_model: LoadModel | None = None,
    predict_video: PredictVideo | None = None,
) -> FastAPI:
    resolved_settings = settings or load_settings()

    app = FastAPI(title="AITRAF Demo Inference API", version="0.1.0")
    app.state.settings = resolved_settings
    app.state.load_model = load_model
    app.state.predict_video = predict_video
    app.include_router(router)
    return app

__all__ = ["create_app"]
