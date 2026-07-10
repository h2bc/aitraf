"""Utilities for downloading referenced clip videos from S3."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from aitraf_train.storage.clips import (
    ClipDownloadRequest,
    ClipDownloadResult,
    download_clips as download_core_clips,
)
from aitraf_core.storage.s3 import build_s3_client, load_s3_settings, parse_s3_uri
from aitraf_train.logging import logger
from aitraf_train.preparation.data_ops import schema
from aitraf_train.preparation.data_ops.utils import strip_clips_prefix


@dataclass
class ClipDownloadConfig:
    """Configuration for downloading S3 clips referenced in the labels file."""

    labels_path: Path | str
    output_dir: Path | str
    force: bool = False

    def __post_init__(self) -> None:
        self.labels_path = Path(self.labels_path)
        self.output_dir = Path(self.output_dir)


def download_clips(config: ClipDownloadConfig) -> None:
    """Download every unique clip referenced in the labels JSONL file."""
    load_dotenv()

    requests = build_clip_download_requests_from_labels(config.labels_path)
    total_clips = len(requests)
    output_dir = config.output_dir
    settings = load_s3_settings(require_bucket=False)
    s3_client = build_s3_client(settings)
    logger.info("Downloading {} clips into {}", total_clips, output_dir)
    counts: Counter[str] = Counter()

    progress_step = max(1, total_clips // 10)

    def on_result(result: ClipDownloadResult, idx: int, total: int) -> None:
        if result.status == "skipped-existing":
            counts.update(skipped=1)
        else:
            counts.update(downloaded=1)
        if idx == total_clips or idx % progress_step == 0:
            pct = (idx / total_clips) * 100
            logger.info(
                "Clip download progress: {}/{} ({:.1f}%)", idx, total_clips, pct
            )

    download_core_clips(
        requests,
        clips_dir=output_dir,
        force=config.force,
        s3_client=s3_client,
        on_result=on_result,
    )

    logger.info(
        "Clip download summary: {} downloaded, {} skipped (total {})",
        counts["downloaded"],
        counts["skipped"],
        total_clips,
    )


def build_clip_download_requests_from_labels(
    labels_path: Path | str,
) -> list[ClipDownloadRequest]:
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise RuntimeError(f"Labels file not found: {labels_path}")

    clip_uris = pd.read_json(labels_path, lines=True)[
        schema.LabelsSchema.video_col
    ].astype(str)
    return [clip_download_request_from_uri(uri) for uri in clip_uris]


def clip_download_request_from_uri(uri: str) -> ClipDownloadRequest:
    _bucket, key = parse_s3_uri(uri)
    relative_key = strip_clips_prefix(Path(key))
    return ClipDownloadRequest(
        video_id=relative_key.as_posix(),
        source_uri=uri,
        relative_path=relative_key,
    )
