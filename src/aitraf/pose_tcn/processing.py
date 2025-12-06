"""Pose TCN processing and batching helpers."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from aitraf.processing import sample_frame_indices


def process_sample(
    sample: dict[str, Any],
    *,
    num_frames: int,
    sampling_dist: str,
    label2id: dict[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Convert a raw dataset sample into TCN-ready tensors."""

    pose_tensor, score_tensor, frame_tensor = _sample_pose_tensor(
        keypoints=sample["keypoints"],
        scores=sample["scores"],
        frames=sample["frames"],
        video_id=sample["video_id"],
        num_frames=num_frames,
        sampling_dist=sampling_dist,
    )

    inputs = pose_tensor.flatten(start_dim=1)
    label_value = torch.tensor(label2id[sample["label"]], dtype=torch.long)

    return {
        "inputs": inputs,
        "labels": label_value,
        "poses": pose_tensor,
        "scores": score_tensor,
        "frames": frame_tensor,
    }


def build_collate(
    *,
    num_frames: int,
    sampling_dist: str,
    label2id: dict[str, int] | None = None,
) -> Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]]:
    """Create a collate_fn that prepares batches for the TCN."""

    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        processed = [
            process_sample(
                sample,
                num_frames=num_frames,
                sampling_dist=sampling_dist,
                label2id=label2id,
            )
            for sample in batch
        ]

        inputs = torch.stack([item["inputs"] for item in processed], dim=0)
        labels = torch.stack([item["labels"] for item in processed], dim=0)
        poses = torch.stack([item["poses"] for item in processed], dim=0)
        scores = torch.stack([item["scores"] for item in processed], dim=0)
        frames = torch.stack([item["frames"] for item in processed], dim=0)

        return {
            "inputs": inputs,
            "labels": labels,
            "poses": poses,
            "scores": scores,
            "frames": frames,
        }

    return _collate


def _sample_pose_tensor(
    *,
    keypoints: np.ndarray,
    scores: np.ndarray,
    frames: np.ndarray,
    video_id: str,
    num_frames: int,
    sampling_dist: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = [
        (
            np.asarray(detections[0], dtype=np.float32),
            np.asarray(score_block[0], dtype=np.float32),
            int(frame_idx),
        )
        for detections, score_block, frame_idx in zip(keypoints, scores, frames)
        if detections.size
    ]

    persons, person_scores, frame_numbers = zip(*valid)

    frame_indices = sample_frame_indices(
        total_frames=len(persons),
        num_frames=num_frames,
        sampling_dist=sampling_dist,
        source=video_id,
    )

    sampled_poses = [
        torch.as_tensor(persons[idx], dtype=torch.float32) for idx in frame_indices
    ]
    sampled_scores = [
        torch.as_tensor(person_scores[idx], dtype=torch.float32)
        for idx in frame_indices
    ]
    sampled_frames = [frame_numbers[idx] for idx in frame_indices]

    return (
        torch.stack(sampled_poses, dim=0),
        torch.stack(sampled_scores, dim=0),
        torch.tensor(sampled_frames, dtype=torch.int64),
    )


__all__ = ["process_sample", "build_collate"]
