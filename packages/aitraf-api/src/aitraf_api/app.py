"""FastAPI application factory."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from aitraf_api.config import Settings, load_settings
from aitraf_api.features import router
from aitraf_api.features.demo_predictions.loader import load_demo_predictions


def create_app(
    *,
    settings: Settings,
) -> FastAPI:
    demo_predictions = load_demo_predictions(settings)

    app = FastAPI(title="AITRAF Demo Predictions API", version="0.1.0")
    app.state.settings = settings
    app.state.demo_predictions = demo_predictions
    app.include_router(router)
    return app


def create_app_from_env() -> FastAPI:
    return create_app(settings=load_settings(env=os.environ, root=Path.cwd()))


__all__ = ["create_app", "create_app_from_env"]
