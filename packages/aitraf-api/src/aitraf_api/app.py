"""FastAPI application factory."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from aitraf_api.config import Settings, load_settings
from aitraf_api.features import router
from aitraf_api.features.demo_predictions.artifacts import (
    PredictionArtifactSource,
    download_prediction_rows,
)
from aitraf_api.features.demo_predictions.service import build_demo_predictions_response


def create_app(
    *,
    settings: Settings,
) -> FastAPI:
    classification_rows = download_prediction_rows(
        PredictionArtifactSource(
            task="trick_classification",
            run_id=settings.classification_predictions_run_id,
        )
    )
    aqa_rows = download_prediction_rows(
        PredictionArtifactSource(
            task="trick_aqa",
            run_id=settings.aqa_predictions_run_id,
        )
    )
    demo_predictions = build_demo_predictions_response(
        classification_rows=classification_rows,
        aqa_rows=aqa_rows,
    )

    app = FastAPI(title="AITRAF Demo Predictions API", version="0.1.0")
    app.state.settings = settings
    app.state.demo_predictions = demo_predictions
    app.include_router(router)
    return app


def create_app_from_env() -> FastAPI:
    return create_app(settings=load_settings(env=os.environ, root=Path.cwd()))


__all__ = ["create_app", "create_app_from_env"]
