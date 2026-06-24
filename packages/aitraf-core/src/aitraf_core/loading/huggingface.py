"""HuggingFace component loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModel


@dataclass(frozen=True)
class HuggingFaceModel:
    model: torch.nn.Module
    processor: Any
    device: str


def load_huggingface_model(
    *,
    model_name: str,
    model_cache_dir: Path | str,
    device: str,
    config_kwargs: Mapping[str, Any] | None = None,
) -> HuggingFaceModel:
    cache_dir = Path(model_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
    )

    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
        **dict(config_kwargs or {}),
    )

    model = AutoModel.from_pretrained(
        model_name,
        config=config,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
    ).to(device)
    model.eval()

    return HuggingFaceModel(
        model=model,
        processor=processor,
        device=device,
    )


__all__ = [
    "HuggingFaceModel",
    "load_huggingface_model",
]
