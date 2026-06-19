"""Pose TCN processing and batching helpers."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from aitraf_core.processing.utils import sample_frame_indices


def process_sample(
    sample: dict[str, Any],
    *,
    num_frames: int,
    sampling_dist: str,
    label_key: str,
    label_transform: Callable[[Any], Any] = lambda x: x,
) -> dict[str, torch.Tensor]:
    """Convert a single-sample task row into TCN-ready tensors."""

    return _build_processed_sample(
        sample,
        num_frames=num_frames,
        sampling_dist=sampling_dist,
        label_value=label_transform(sample[label_key]),
    )


def _build_processed_sample(
    sample: dict[str, Any],
    *,
    num_frames: int,
    sampling_dist: str,
    label_value: Any,
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
    label_tensor = torch.as_tensor(label_value)

    return {
        "inputs": inputs,
        "labels": label_tensor,
        "poses": pose_tensor,
        "scores": score_tensor,
        "frames": frame_tensor,
    }


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


__all__ = ["process_sample"]
