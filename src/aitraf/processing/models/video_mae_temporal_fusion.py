"""VideoMAE input processing for temporal-fusion models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from transformers import VideoMAEImageProcessor

from aitraf.processing.video import load_segmented_video_frames


def process_temporal_fusion_sample(
    manifest_row: dict[str, Any],
    processor: VideoMAEImageProcessor,
    local_clips_dir: str | Path,
    num_frames: int,
    num_clips: int,
    sampling_dist: str,
    label_key: str,
    label_transform: Callable[[Any], Any] = lambda x: x,
) -> dict[str, torch.Tensor]:
    """Load one video as equal temporal clips for a temporal-fusion model."""

    return {
        "pixel_values": process_temporal_fusion_video(
            video_id=manifest_row["video_id"],
            processor=processor,
            local_clips_dir=local_clips_dir,
            num_frames=num_frames,
            num_clips=num_clips,
            sampling_dist=sampling_dist,
        ),
        "labels": torch.as_tensor(label_transform(manifest_row[label_key])),
    }


def process_temporal_fusion_video(
    *,
    video_id: str,
    processor: VideoMAEImageProcessor,
    local_clips_dir: str | Path,
    num_frames: int,
    num_clips: int,
    sampling_dist: str,
) -> torch.Tensor:
    segments, _ = load_segmented_video_frames(
        video_id=video_id,
        local_clips_dir=local_clips_dir,
        num_segments=num_clips,
        frames_per_segment=num_frames,
        sampling_dist=sampling_dist,
    )
    clip_pixel_values = [
        processor(frames, return_tensors="pt")["pixel_values"][0]
        for frames in segments
    ]
    return torch.stack(clip_pixel_values)


class VideoMaeTemporalFusionClassifier(nn.Module):
    """Encode temporal clips with VideoMAE, then fuse clip embeddings."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        hidden_size: int,
        num_labels: int,
        num_clips: int,
        fusion_layers: int,
        fusion_heads: int,
        fusion_dropout: float,
        loss_fn: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.clip_norm = nn.LayerNorm(hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_clips + 1, hidden_size)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=fusion_heads,
            dim_feedforward=hidden_size * 4,
            dropout=fusion_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            encoder_layer,
            num_layers=fusion_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = loss_fn

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_clips = pixel_values.shape[:2]
        clip_shape = pixel_values.shape[2:]
        flat_pixel_values = pixel_values.reshape(batch_size * num_clips, *clip_shape)

        outputs = self.encoder(pixel_values=flat_pixel_values)
        clip_embeddings = outputs.last_hidden_state.mean(dim=1)
        clip_embeddings = self.clip_norm(clip_embeddings)
        clip_embeddings = clip_embeddings.reshape(batch_size, num_clips, -1)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, clip_embeddings], dim=1)
        tokens = tokens + self.position_embeddings[:, : num_clips + 1]
        fused_embedding = self.fusion(tokens)[:, 0]
        logits = self.classifier(fused_embedding)

        result = {"logits": logits}
        if labels is not None:
            result["loss"] = self.loss_fn(logits, labels.long())
        return result


__all__ = [
    "VideoMaeTemporalFusionClassifier",
    "process_temporal_fusion_sample",
    "process_temporal_fusion_video",
]
