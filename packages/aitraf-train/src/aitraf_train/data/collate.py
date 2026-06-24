"""Collate helpers for train/eval datasets."""

from __future__ import annotations

from typing import Any, Callable

import torch


def build_collate(
    process_sample: Callable[..., dict[str, torch.Tensor]],
    **process_kwargs: Any,
) -> Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]]:
    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        processed = [process_sample(sample, **process_kwargs) for sample in batch]

        return {
            key: torch.stack([item[key] for item in processed], dim=0)
            for key in processed[0]
        }

    return _collate


__all__ = ["build_collate"]
