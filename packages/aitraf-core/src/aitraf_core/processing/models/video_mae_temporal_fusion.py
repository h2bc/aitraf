"""VideoMAE input processing for temporal-fusion models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from aitraf_core.utils.huggingface import hf_model_cache_dir_name


def video_feature_cache_subdir(
    *,
    backbone: str,
    num_clips: int,
    sample_frames: int,
    sampling_dist: str,
) -> Path:
    config_slug = f"clips_{num_clips}_frames_{sample_frames}_sampling_{sampling_dist}"
    return Path(hf_model_cache_dir_name(backbone)) / config_slug


def video_feature_cache_path(
    *,
    features_dir: Path | str,
    video_id: str,
    backbone: str,
    num_clips: int,
    sample_frames: int,
    sampling_dist: str,
) -> Path:
    relative_video = Path(video_id)
    relative_feature = relative_video.with_suffix(".pt")
    return (
        Path(features_dir)
        / video_feature_cache_subdir(
            backbone=backbone,
            num_clips=num_clips,
            sample_frames=sample_frames,
            sampling_dist=sampling_dist,
        )
        / relative_feature
    )


def process_temporal_fusion_feature_sample(
    manifest_row: dict[str, Any],
    features_dir: str | Path,
    backbone: str,
    num_frames: int,
    num_clips: int,
    sampling_dist: str,
    label_key: str,
    label_transform: Callable[[Any], Any] = lambda x: x,
) -> dict[str, torch.Tensor]:
    """Load one cached temporal-fusion feature sequence and label."""

    feature_path = video_feature_cache_path(
        features_dir=features_dir,
        video_id=manifest_row["video_id"],
        backbone=backbone,
        num_clips=num_clips,
        sample_frames=num_frames,
        sampling_dist=sampling_dist,
    )
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing cached VideoMAE features for {manifest_row['video_id']}: "
            f"{feature_path}. Run data_ops.video_mae_feature_extraction first."
        )

    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    return {
        "features": payload["features"].float(),
        "labels": torch.as_tensor(label_transform(manifest_row[label_key])),
    }


class VideoMaeTemporalFusionClassifier(nn.Module):
    """Decode class queries over cached temporal clip features."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_labels: int,
        num_clips: int,
        num_queries: int | None = None,
        fusion_layers: int,
        fusion_heads: int,
        fusion_dropout: float,
        query_init_std: float,
        loss_fn: nn.Module,
    ) -> None:
        super().__init__()
        query_count = num_queries or num_clips
        self.clip_norm = nn.LayerNorm(hidden_size)
        self.query_embeddings = nn.Parameter(torch.zeros(1, query_count, hidden_size))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=fusion_heads,
            dim_feedforward=hidden_size * 4,
            dropout=fusion_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerDecoder(
            decoder_layer,
            num_layers=fusion_layers,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(query_count * hidden_size),
            nn.Linear(query_count * hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(hidden_size, num_labels),
        )
        self.loss_fn = loss_fn

        nn.init.trunc_normal_(self.query_embeddings, std=query_init_std)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = features.shape[0]
        clip_embeddings = self.clip_norm(features)

        memory = clip_embeddings
        queries = self.query_embeddings.expand(batch_size, -1, -1)
        query_embeddings = self.fusion(tgt=queries, memory=memory)
        logits = self.classifier(query_embeddings.flatten(start_dim=1))

        result = {"logits": logits}
        if labels is not None:
            result["loss"] = self.loss_fn(logits, labels.long())
        return result


__all__ = [
    "VideoMaeTemporalFusionClassifier",
    "process_temporal_fusion_feature_sample",
    "video_feature_cache_path",
    "video_feature_cache_subdir",
]
