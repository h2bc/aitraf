"""Frame sampling helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch


def sample_frame_indices(
    frame_range: tuple[int, int],
    num_frames: int,
    sampling_dist: str,
    *,
    source: Path | str | None = None,
) -> List[int]:
    start, end = frame_range
    total_frames = end - start
    if total_frames <= 0:
        context = f" ({source})" if source is not None else ""
        raise ValueError(f"Invalid frame range {frame_range}{context}.")

    if total_frames < num_frames:
        indices = torch.linspace(0, total_frames - 1, steps=num_frames).long().tolist()
        return [start + idx for idx in indices]

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

    return [start + idx for idx in indices]


def _gaussian_indices(total_frames: int, num_frames: int) -> List[int]:
    if num_frames == 1:
        return [int((total_frames - 1) / 2)]

    positions = torch.linspace(-1, 1, steps=total_frames, dtype=torch.float32)
    weights = torch.exp(-4 * positions.square())
    probs = weights / weights.sum()
    indices = torch.multinomial(probs, num_frames, replacement=False)
    indices, _ = torch.sort(indices)

    return indices.long().tolist()


def _center_indices(total_frames: int, num_frames: int) -> List[int]:
    start = (total_frames - num_frames) // 2
    return list(range(start, start + num_frames))


__all__ = ["sample_frame_indices"]
