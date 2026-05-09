"""Shared processing utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, List

import torch


def load_target_label_mappings(
    vocab_path: Path | str, column_name: str
) -> tuple[list[str], dict[str, int], dict[str, str]]:
    """Load label/id mappings emitted by the manifest step."""

    vocab_path = Path(vocab_path)

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    with vocab_path.open(encoding="utf-8") as fh:
        labels_config = json.load(fh)

    labels = labels_config[column_name]["labels"]
    label2id = labels_config[column_name]["label2id"]
    id2label = labels_config[column_name]["id2label"]

    return labels, label2id, id2label


def sample_frame_indices(
    total_frames: int,
    num_frames: int,
    sampling_dist: str,
    *,
    source: Path | str | None = None,
) -> List[int]:
    """Return ordered frame indices sampled from a sequence."""
    if total_frames < num_frames:
        context = f" ({source})" if source is not None else ""
        raise ValueError(
            f"Sequence{context} has only {total_frames} frames (<{num_frames})."
        )

    if sampling_dist == "gaussian":
        indices = _gaussian_indices(total_frames, num_frames)
    elif sampling_dist == "center":
        indices = _center_indices(total_frames, num_frames)
    elif sampling_dist == "uniform":
        indices = torch.linspace(0, total_frames - 1, steps=num_frames).long().tolist()
    else:
        raise ValueError(
            f"Unsupported sampling_dist '{sampling_dist}'. "
            "Expected 'gaussian', 'center', or 'uniform'."
        )

    return indices


def build_collate(
    process_sample: Callable[..., dict[str, torch.Tensor]],
    **process_kwargs: Any,
) -> Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]]:
    """Create a collate_fn from a per-sample processing callback."""

    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        processed = [process_sample(sample, **process_kwargs) for sample in batch]

        return {
            k: torch.stack([item[k] for item in processed], dim=0) for k in processed[0]
        }

    return _collate


def _gaussian_indices(total_frames: int, num_frames: int) -> List[int]:
    """Sample indices with higher probability near the sequence center."""

    if num_frames == 1:
        return [int((total_frames - 1) / 2)]

    positions = torch.linspace(-1, 1, steps=total_frames, dtype=torch.float32)
    weights = torch.exp(-4 * positions.square())
    probs = weights / weights.sum()
    indices = torch.multinomial(probs, num_frames, replacement=False)
    indices, _ = torch.sort(indices)

    return indices.long().tolist()


def _center_indices(total_frames: int, num_frames: int) -> List[int]:
    """Return the deterministic centered frame window."""

    start = (total_frames - num_frames) // 2
    return list(range(start, start + num_frames))


__all__ = ["build_collate", "load_target_label_mappings", "sample_frame_indices"]
