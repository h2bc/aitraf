"""Utilities for downloading referenced clip videos from S3."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from aitraf.data import schema
from aitraf.logging import logger


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
    labels_path = config.labels_path
    if not labels_path.exists():
        raise RuntimeError(f"Labels file not found: {labels_path}")

    clip_uris = _collect_clip_uris(labels_path)
    total_clips = len(clip_uris)
    if not total_clips:
        logger.info("No clip URIs found in {}", labels_path)
        return

    endpoint_url = os.getenv("AWS_ENDPOINT_URL")
    region_name = os.getenv("AWS_DEFAULT_REGION")
    if not endpoint_url or not region_name:
        raise RuntimeError(
            "AWS_ENDPOINT_URL and AWS_DEFAULT_REGION must be set in the environment/.env"
        )

    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region_name,
    )
    output_dir = config.output_dir
    logger.info("Downloading {} clips into {}", total_clips, output_dir)
    success_count = 0
    failure_count = 0
    skipped_count = 0

    progress_step = max(1, total_clips // 10)

    for idx, uri in enumerate(sorted(clip_uris), start=1):
        bucket, key = _parse_s3_uri(uri)
        relative_key = _strip_prefix(Path(key))
        destination = output_dir / relative_key
        if destination.exists() and not config.force:
            skipped_count += 1
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            s3_client.download_file(bucket, key, str(destination))
        except ClientError as exc:  # pragma: no cover - network path log
            failure_count += 1
            logger.warning("Failed to download {}: {}", uri, exc)
        else:
            success_count += 1
            if idx == total_clips or idx % progress_step == 0:
                pct = (idx / total_clips) * 100
                logger.info(
                    "Clip download progress: {}/{} ({:.1f}%)", idx, total_clips, pct
                )

    logger.info(
        "Clip download summary: {} downloaded, {} skipped, {} failed (total {})",
        success_count,
        skipped_count,
        failure_count,
        total_clips,
    )


def _collect_clip_uris(labels_path: Path) -> set[str]:
    uris: set[str] = set()
    with labels_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            clip_path = record.get(schema.VIDEO_COLUMN)
            if isinstance(clip_path, str) and clip_path.startswith("s3://"):
                uris.add(clip_path)
    return uris


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Unsupported clip URI: {uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"Invalid S3 key in URI: {uri}")
    return bucket, key


def _strip_prefix(path: Path) -> Path:
    """Drop a leading 'clips/' prefix so downloads land under data/clips."""
    parts = path.parts
    if parts and parts[0] == "clips":
        parts = parts[1:]
    if not parts:
        return Path(path.name)
    return Path(*parts)
