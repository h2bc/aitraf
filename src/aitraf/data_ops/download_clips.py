"""Utilities for downloading referenced clip videos from S3."""

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from aitraf.logging import logger
from aitraf.data_ops.utils import strip_clips_prefix
from aitraf.data_ops import schema
import pandas as pd


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

    clip_uris = pd.read_json(labels_path, lines=True)[
        schema.LabelsSchema.input_col
    ].astype(str)
    total_clips = len(clip_uris)

    output_dir = config.output_dir
    s3_client = _build_s3_client()
    logger.info("Downloading {} clips into {}", total_clips, output_dir)
    success_count = 0
    failure_count = 0
    skipped_count = 0

    progress_step = max(1, total_clips // 10)

    for idx, uri in enumerate(clip_uris, start=1):
        bucket, key = _parse_s3_uri(uri)
        relative_key = strip_clips_prefix(Path(key))
        destination = output_dir / relative_key
        if destination.exists() and not config.force:
            skipped_count += 1
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            s3_client.download_file(bucket, key, str(destination))
        except ClientError as exc:
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


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Unsupported clip URI: {uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"Invalid S3 key in URI: {uri}")
    return bucket, key


def _build_s3_client():
    endpoint_url, region_name, access_key, secret_key = _load_required_aws_settings()
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def _load_required_aws_settings() -> tuple[str, str, str, str]:
    settings = {
        "AWS_ENDPOINT_URL": os.getenv("AWS_ENDPOINT_URL"),
        "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION"),
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    }
    missing = [name for name, value in settings.items() if not value]
    if missing:
        raise RuntimeError(
            "The following AWS environment variables must be set: " + ", ".join(missing)
        )

    return (
        settings["AWS_ENDPOINT_URL"],
        settings["AWS_DEFAULT_REGION"],
        settings["AWS_ACCESS_KEY_ID"],
        settings["AWS_SECRET_ACCESS_KEY"],
    )
