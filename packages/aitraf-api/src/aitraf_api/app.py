"""FastAPI application factory."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI

from aitraf_api.config import Settings, load_settings
from aitraf_api.features import router
from aitraf_api.features.demo_predictions.artifacts import download_demo_prediction_rows
from aitraf_api.features.demo_predictions.thumbnails import (
    ensure_prediction_thumbnails,
    find_rows_with_missing_thumbnails,
)
from aitraf_api.features.demo_predictions.service import match_prediction_rows
from aitraf_api.features.demo_predictions.videos import (
    create_asset_url_presigner,
    download_prediction_videos,
)


def create_app(
    *,
    settings: Settings,
) -> FastAPI:
    classification_prediction_rows, aqa_prediction_rows = (
        download_demo_prediction_rows(settings)
    )
    classification_prediction_rows, aqa_prediction_rows = match_prediction_rows(
        classification_prediction_rows,
        aqa_prediction_rows,
    )
    rows_missing_thumbnails = find_rows_with_missing_thumbnails(
        classification_prediction_rows,
        aws_endpoint_url=settings.aws_endpoint_url,
        aws_bucket=settings.aws_bucket,
    )
    with tempfile.TemporaryDirectory(prefix="aitraf-demo-videos-") as temp_dir:
        local_videos = download_prediction_videos(
            rows_missing_thumbnails,
            aws_endpoint_url=settings.aws_endpoint_url,
            aws_bucket=settings.aws_bucket,
            directory=Path(temp_dir),
        )
        classification_prediction_rows = ensure_prediction_thumbnails(
            classification_prediction_rows,
            local_videos=local_videos,
            aws_endpoint_url=settings.aws_endpoint_url,
            aws_bucket=settings.aws_bucket,
        )
    presign_asset_url = create_asset_url_presigner(
        aws_endpoint_url=settings.aws_endpoint_url,
        aws_bucket=settings.aws_bucket,
    )

    app = FastAPI(title="AITRAF Demo Predictions API", version="0.1.0")
    app.state.settings = settings
    app.state.classification_prediction_rows = classification_prediction_rows
    app.state.aqa_prediction_rows = aqa_prediction_rows
    app.state.presign_asset_url = presign_asset_url
    app.include_router(router)
    return app


def create_app_from_env() -> FastAPI:
    return create_app(settings=load_settings(env=os.environ, root=Path.cwd()))


__all__ = ["create_app", "create_app_from_env"]
