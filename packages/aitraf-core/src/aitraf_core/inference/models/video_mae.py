"""VideoMAE inference helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import VideoMAEImageProcessor

from aitraf_core.inference.classification import decode_classification_logits
from aitraf_core.processing.models.video_mae import process_video
from aitraf_core.utils import LoadedTransformersModel


def predict_video_mae_logits(
    *,
    model: nn.Module,
    processor: VideoMAEImageProcessor,
    video_id: str,
    local_clips_dir: str | Path,
    num_frames: int,
    sampling_dist: str,
) -> torch.Tensor:
    model.eval()
    pixel_values = process_video(
        video_id=video_id,
        processor=processor,
        local_clips_dir=local_clips_dir,
        num_frames=num_frames,
        sampling_dist=sampling_dist,
    )
    with torch.no_grad():
        output = model(pixel_values=pixel_values.unsqueeze(0))
    return output.logits[0].float()


def predict_video_mae_label(
    *,
    loaded_model: LoadedTransformersModel,
    video_id: str,
    local_clips_dir: str | Path,
) -> tuple[str, float]:
    model = loaded_model.model
    logits = predict_video_mae_logits(
        model=model,
        processor=loaded_model.image_processor,
        video_id=video_id,
        local_clips_dir=local_clips_dir,
        num_frames=int(model.config.num_frames),
        sampling_dist=loaded_model.run_params["train_sampling_dist"],
    )
    id2label = {str(key): value for key, value in model.config.id2label.items()}
    return decode_classification_logits(logits, id2label)


__all__ = [
    "predict_video_mae_label",
    "predict_video_mae_logits",
]
