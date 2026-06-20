"""Temporal-fusion inference helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from aitraf_core.inference.classification import decode_classification_logits


def predict_temporal_fusion_logits(
    *,
    model: nn.Module,
    features: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    batched_features = features if features.ndim == 3 else features.unsqueeze(0)
    with torch.inference_mode():
        output = model(features=batched_features)

    logits = _extract_logits(output)
    if logits.ndim == 2:
        return logits[0].float()
    if logits.ndim == 1:
        return logits.float()
    raise ValueError(f"Unsupported temporal-fusion logits shape: {tuple(logits.shape)}")


def predict_temporal_fusion_label(
    *,
    model: nn.Module,
    features: torch.Tensor,
    id2label: dict[str, str] | dict[int, str],
) -> tuple[str, float]:
    logits = predict_temporal_fusion_logits(model=model, features=features)
    normalized_id2label = {str(key): str(value) for key, value in id2label.items()}
    return decode_classification_logits(logits, normalized_id2label)


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict) and "logits" in output:
        return torch.as_tensor(output["logits"])
    logits = getattr(output, "logits", None)
    if logits is not None:
        return torch.as_tensor(logits)
    raise ValueError("Temporal-fusion model output missing logits")


__all__ = [
    "predict_temporal_fusion_label",
    "predict_temporal_fusion_logits",
]
