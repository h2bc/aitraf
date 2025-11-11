"""Utilities for converting manifest entries into VideoMAE-ready tensors."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

import av
import boto3
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import VideoMAEImageProcessor
from dotenv import load_dotenv

from aitraf import dataset_schema

load_dotenv()


def load_manifest(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"Manifest {path} contains no rows")
    return records


def load_label_mapping(labels_path: Path, column: str) -> dict[str, int]:
    with labels_path.open() as fh:
        meta = json.load(fh)
    label_info = meta[column]
    return {name: int(idx) for name, idx in label_info["label2id"].items()}


def uniform_indices(total_frames: int, target_frames: int) -> list[int]:
    if total_frames <= 0 or target_frames <= 0:
        return [0] * max(target_frames, 1)
    if target_frames == 1:
        return [min(total_frames - 1, 0)]
    positions = np.linspace(0, total_frames - 1, num=target_frames)
    return [int(round(pos)) for pos in positions]


@dataclass
class ClipSampling:
    num_frames: int = 16
    frame_size: int = 224
    rotation_quadrants: int = 0  # 0-3


class ClipDataset(Dataset):
    """Minimal dataset for decoding clips and preparing VideoMAE tensors."""

    def __init__(
        self,
        records: list[dict],
        label2id: dict[str, int],
        processor: VideoMAEImageProcessor,
        sampling: ClipSampling | None = None,
        *,
        video_column: str = dataset_schema.VIDEO_COLUMN,
        label_column: str = dataset_schema.TARGET_COLUMN,
    ) -> None:
        self.records = records
        self.label2id = label2id
        self.processor = processor
        self.sampling = sampling or ClipSampling()
        self.video_column = video_column
        self.label_column = label_column
        self._s3: boto3.client | None = None

    def __len__(self) -> int:  # pragma: no cover trivial
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | Sequence[str]]:
        record = self.records[idx]
        frames = self._decode_frames(record[self.video_column])
        rotated = [self._rotate_frame(frame) for frame in frames]
        time_indices = uniform_indices(len(rotated), self.sampling.num_frames)
        sampled_frames = [rotated[i] for i in time_indices]
        processed = self.processor(
            sampled_frames,
            return_tensors="pt",
            size={"height": self.sampling.frame_size, "width": self.sampling.frame_size},
        )
        # processor returns [batch, T, C, H, W]; squeeze batch -> [T, C, H, W]
        pixel_values = processed["pixel_values"].squeeze(0)
        label_name = record[self.label_column]
        label_id = self.label2id[label_name]
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label_id, dtype=torch.long),
            "label_name": label_name,
            "video_path": record[self.video_column],
        }

    def _decode_frames(self, uri: str) -> list[np.ndarray]:
        local_path, cleanup = self._materialize(uri)
        try:
            container = av.open(str(local_path))
            frames = [frame.to_rgb().to_ndarray() for frame in container.decode(video=0)]
            container.close()
            if not frames:
                raise RuntimeError(f"No frames decoded from {local_path}")
            return frames
        finally:
            if cleanup:
                local_path.unlink(missing_ok=True)

    def _rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        k = self.sampling.rotation_quadrants % 4
        if k == 0:
            return frame
        return np.rot90(frame, k=k)

    def _materialize(self, uri: str) -> tuple[Path, bool]:
        if uri.startswith("s3://"):
            if self._s3 is None:
                self._s3 = boto3.client("s3")
            parsed = urlparse(uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            suffix = Path(key).suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp_path = Path(tmp.name)
            tmp.close()
            with tmp_path.open("wb") as fh:
                self._s3.download_fileobj(bucket, key, fh)
            return tmp_path, True
        path = Path(uri)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        return path, False


def build_clip_dataset(
    manifest_path: str | Path,
    labels_path: str | Path,
    *,
    processor_name: str = "OpenGVLab/VideoMAEv2-Base",
    sampling: ClipSampling | None = None,
) -> ClipDataset:
    records = load_manifest(Path(manifest_path))
    label2id = load_label_mapping(Path(labels_path), dataset_schema.TARGET_COLUMN)
    processor = VideoMAEImageProcessor.from_pretrained(processor_name)
    return ClipDataset(records, label2id, processor, sampling)


__all__ = ["ClipDataset", "ClipSampling", "build_clip_dataset", "load_manifest"]
