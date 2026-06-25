"""Label helpers for train/eval datasets."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


@dataclass(frozen=True)
class TargetLabelMappings:
    target_column: str
    labels: list[str]
    label2id: dict[str, int]
    id2label: dict[str, str]

    def __iter__(self) -> Iterator[Any]:
        yield self.labels
        yield self.label2id
        yield self.id2label

    def model_metadata(self) -> dict[str, Any]:
        return {
            "target_column": self.target_column,
            "labels": self.labels,
            "label2id": self.label2id,
            "id2label": self.id2label,
        }


def load_target_label_mappings(
    vocab_path: Path | str, column_name: str
) -> TargetLabelMappings:
    vocab_path = Path(vocab_path)

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    with vocab_path.open(encoding="utf-8") as fh:
        labels_config = json.load(fh)

    labels = labels_config[column_name]["labels"]
    label2id = labels_config[column_name]["label2id"]
    id2label = labels_config[column_name]["id2label"]

    return TargetLabelMappings(
        target_column=column_name,
        labels=labels,
        label2id=label2id,
        id2label=id2label,
    )


def build_label_transform(label2id: dict[str, int]) -> Callable[[Any], int]:
    return lambda label: label2id[str(label)]


def build_class_weights(
    target_ids: list[int],
    *,
    num_labels: int,
    device: str,
) -> torch.Tensor:
    target_ids = np.asarray(target_ids)
    classes = np.arange(num_labels)
    missing_classes = sorted(set(classes) - set(target_ids))

    if missing_classes:
        raise ValueError(
            "Cannot use balanced class weights when a class has zero train samples: "
            f"{missing_classes}"
        )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=target_ids,
    )
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


__all__ = [
    "TargetLabelMappings",
    "build_class_weights",
    "build_label_transform",
    "load_target_label_mappings",
]
