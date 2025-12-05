"""Shared helpers for VideoMAE training and evaluation."""

from functools import reduce
from pathlib import Path
from typing import Callable

import torch

from aitraf.video_mae.processing import process_clip


def build_collate(
    processor,
    clips_dir: Path | str,
    label2id: dict[str, int],
    sample_frames: int,
    sampling_dist: str,
) -> Callable:
    """Create a collate_fn consistent across training and eval."""

    def _collate(batch):
        processed_batch = [
            process_clip(
                row,
                processor,
                clips_dir,
                label2id,
                sample_frames,
                sampling_dist,
            )
            for row in batch
        ]

        pivot = reduce(
            lambda acc, x: {k: acc.get(k, []) + [x[k]] for k in x},
            processed_batch,
            {},
        )

        return {k: torch.stack(v) for k, v in pivot.items()}

    return _collate
