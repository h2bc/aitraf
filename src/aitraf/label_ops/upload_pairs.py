"""Upload generated pairwise labeling tasks to S3 under the `pairs/` prefix."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from aitraf.logging import logger


@dataclass
class PairUploadConfig:
    """Configuration for uploading generated pair files to S3."""

    pairs_dir: Path | str
    force: bool = False

    def __post_init__(self) -> None:
        self.pairs_dir = Path(self.pairs_dir)


def upload_pairs(config: PairUploadConfig) -> int:
    """Upload pair files to `s3://$AWS_BUCKET/pairs/`."""
    load_dotenv()

    pairs_dir = config.pairs_dir
    if not pairs_dir.exists():
        raise RuntimeError(f"Pairs directory not found: {pairs_dir}")
    if not pairs_dir.is_dir():
        raise RuntimeError(f"Pairs path is not a directory: {pairs_dir}")

    endpoint_url, region_name, access_key, secret_key, bucket = (
        _load_required_aws_settings()
    )

    files = sorted(p for p in pairs_dir.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"No pair files found under {pairs_dir}")

    s3_client = _build_s3_client(
        endpoint_url=endpoint_url,
        region_name=region_name,
        access_key=access_key,
        secret_key=secret_key,
    )

    uploaded = 0
    skipped = 0
    failed = 0

    logger.info(
        "Uploading {} pair files to s3://{}/pairs/ (force={})",
        len(files),
        bucket,
        config.force,
    )

    progress_step = max(1, len(files) // 10)

    for idx, path in enumerate(files, start=1):
        rel = path.relative_to(pairs_dir)
        key = f"pairs/{rel.as_posix()}"

        if not config.force and _object_exists(s3_client, bucket=bucket, key=key):
            skipped += 1
            continue

        try:
            s3_client.upload_file(str(path), bucket, key)
        except ClientError as exc:
            failed += 1
            logger.warning(
                "Failed to upload {} -> s3://{}/{}: {}", path, bucket, key, exc
            )
        else:
            uploaded += 1
            if idx == len(files) or idx % progress_step == 0:
                pct = (idx / len(files)) * 100
                logger.info(
                    "Pair upload progress: {}/{} ({:.1f}%)",
                    idx,
                    len(files),
                    pct,
                )

    logger.info(
        "Pair upload summary: {} uploaded, {} skipped, {} failed (total {})",
        uploaded,
        skipped,
        failed,
        len(files),
    )

    if uploaded == 0 and not skipped:
        raise RuntimeError("No pair files were uploaded.")

    return uploaded


def _object_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    else:
        return True


def _build_s3_client(
    *,
    endpoint_url: str,
    region_name: str,
    access_key: str,
    secret_key: str,
):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def _load_required_aws_settings() -> tuple[str, str, str, str, str]:
    settings = {
        "AWS_ENDPOINT_URL": os.getenv("AWS_ENDPOINT_URL"),
        "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION"),
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "AWS_BUCKET": os.getenv("AWS_BUCKET"),
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
        settings["AWS_BUCKET"],
    )
