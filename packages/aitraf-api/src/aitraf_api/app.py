"""FastAPI application factory."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from aitraf_core.loading import (
    load_huggingface_model,
    load_mlflow_torch_model,
    load_mlflow_transformers_model,
)
from aitraf_core.pre_processing import video_feature_cache_dir
from aitraf_api.config import (
    Settings,
    TrickAssessmentPreProcessingConfig,
    load_settings,
)
from aitraf_api.features import router
from aitraf_api.features.demo_videos.download import hydrate_demo_clips


def create_app(
    *,
    settings: Settings,
) -> FastAPI:
    hydrate_demo_clips(settings)

    classification_model = load_mlflow_transformers_model(
        settings.classification.model_uri,
        device=settings.device,
    )

    aqa_model = load_mlflow_torch_model(
        settings.aqa.model_uri,
        device=settings.device,
    )
    aqa_params = aqa_model.run_params
    aqa_num_clips = int(aqa_params["num_clips"])
    aqa_sample_frames = int(aqa_params["num_frames"])
    aqa_sampling_dist = aqa_params["train_sampling_dist"]
    aqa_metadata = aqa_model.metadata
    if aqa_metadata is None:
        raise RuntimeError(
            "AQA model metadata is missing. Retrain and re-register the model."
        )
    aqa_id2label = aqa_metadata.get("id2label")
    if not isinstance(aqa_id2label, dict):
        raise RuntimeError(
            "AQA model metadata is missing id2label. Retrain and re-register the model."
        )
    if aqa_metadata.get("target_column") != settings.aqa.ground_truth_field:
        raise RuntimeError(
            "AQA model metadata target_column does not match API ground truth field."
        )
    aqa_pre_processing = TrickAssessmentPreProcessingConfig(
        backbone=aqa_params["backbone"],
        num_clips=aqa_num_clips,
        sample_frames=aqa_sample_frames,
        sampling_dist=aqa_sampling_dist,
        id2label=aqa_id2label,
        feature_cache_dir=video_feature_cache_dir(
            features_dir=settings.aqa.features_dir,
            backbone=aqa_params["backbone"],
            num_clips=aqa_num_clips,
            sample_frames=aqa_sample_frames,
            sampling_dist=aqa_sampling_dist,
        ),
    )
    aqa_feature_extractor = load_huggingface_model(
        model_name=aqa_pre_processing.backbone,
        model_cache_dir=settings.aqa.model_cache_dir,
        device=settings.device,
        config_kwargs={"num_frames": aqa_pre_processing.sample_frames},
    )

    app = FastAPI(title="AITRAF Demo Inference API", version="0.1.0")
    app.state.settings = settings
    app.state.classification_model = classification_model
    app.state.aqa_model = aqa_model
    app.state.aqa_pre_processing = aqa_pre_processing
    app.state.aqa_feature_extractor = aqa_feature_extractor
    app.include_router(router)
    return app


def create_app_from_env() -> FastAPI:
    return create_app(settings=load_settings(env=os.environ, root=Path.cwd()))


__all__ = ["create_app", "create_app_from_env"]
