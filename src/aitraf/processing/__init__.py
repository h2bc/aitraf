"""Shared processing helpers for model pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch

from aitraf.data_ops import schema


def load_target_label_mappings(
    manifests_dir: Path | str,
) -> tuple[list[str], dict[str, int], dict[str, str]]:
    """Load label/id mappings emitted by the manifest step."""

    manifests_dir = Path(manifests_dir)
    labels_path = manifests_dir / "labels.json"

    with labels_path.open(encoding="utf-8") as fh:
        labels_config = json.load(fh)

    labels = labels_config[schema.TARGET_COLUMN]["labels"]
    label2id = labels_config[schema.TARGET_COLUMN]["label2id"]
    id2label = labels_config[schema.TARGET_COLUMN]["id2label"]

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

    if sampling_dist == "gaussian_stochastic":
        indices = _gaussian_center_indices(total_frames, num_frames)
    elif sampling_dist == "uniform":
        indices = torch.linspace(0, total_frames - 1, steps=num_frames).long().tolist()
    else:
        raise ValueError(
            f"Unsupported sampling_dist '{sampling_dist}'. "
            "Expected 'uniform' or 'gaussian_stochastic'."
        )

    return indices


def _gaussian_center_indices(total_frames: int, num_frames: int) -> List[int]:
    """Sample indices with higher probability near the sequence center."""

    if num_frames == 1:
        return [int((total_frames - 1) / 2)]

    positions = torch.linspace(-1, 1, steps=total_frames, dtype=torch.float32)
    weights = torch.exp(-4 * positions.square())
    probs = weights / weights.sum()
    indices = torch.multinomial(probs, num_frames, replacement=False)
    indices, _ = torch.sort(indices)

    return indices.long().tolist()


__all__ = ["load_target_label_mappings", "sample_frame_indices"]
