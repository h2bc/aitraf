"""Shared helpers for VideoMAE training and evaluation."""

from functools import reduce
import json
from pathlib import Path
from typing import Any, Callable

import torch

from aitraf.data import schema
from aitraf.video_mae.processing import process_clip


def load_target_label_mappings(
    manifests_dir: Path | str,
) -> tuple[dict[str, int], dict[str, str]]:
    """Load label/id mappings from the labels manifest."""

    labels_config = _read_json(Path(manifests_dir) / "labels.json")

    labels = labels_config[schema.TARGET_COLUMN]["labels"]
    label2id = labels_config[schema.TARGET_COLUMN]["label2id"]
    id2label = labels_config[schema.TARGET_COLUMN]["id2label"]

    return labels, label2id, id2label


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


def _read_json(json_path: Path) -> dict[str, Any]:
    with open(json_path) as f:
        data = json.load(f)

    return data
