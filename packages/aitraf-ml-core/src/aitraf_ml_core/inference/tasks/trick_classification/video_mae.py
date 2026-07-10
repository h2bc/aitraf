"""VideoMAE trick-classification inference."""

from __future__ import annotations

from pathlib import Path

from aitraf_ml_core.inference.models.video_mae import (
    predict as predict_label,
    predict_feature_vectors,
)
from aitraf_ml_core.processing.models.video_mae import process_video_mae_clip
from aitraf_ml_core.loading import MlflowTransformersModel


def predict_trick_classification_video_mae(
    *,
    loaded_model: MlflowTransformersModel,
    video_id: str,
    local_clips_dir: Path,
) -> tuple[str, float]:
    clip_input = process_video_mae_clip(
        video_id=video_id,
        processor=loaded_model.image_processor,
        local_clips_dir=local_clips_dir,
        num_frames=int(loaded_model.model.config.num_frames),
        sampling_dist=loaded_model.run_params["train_sampling_dist"],
    )

    feature_vectors = predict_feature_vectors(
        model=loaded_model.model,
        inputs=clip_input.unsqueeze(0),
    )

    return predict_label(
        classifier_head=loaded_model.model.classifier,
        feature_vectors=feature_vectors,
        id2label=loaded_model.model.config.id2label,
    )


__all__ = ["predict_trick_classification_video_mae"]
