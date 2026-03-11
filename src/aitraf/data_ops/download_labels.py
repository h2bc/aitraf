"""Download label annotations from S3 and merge into one JSONL file."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from aitraf.logging import logger
from aitraf.utils.s3_utils import build_s3_client, iter_keys, load_s3_settings


@dataclass
class LabelDownloadConfig:
    """Configuration for downloading and merging label annotations from S3."""

    prefix: str
    output_path: Path | str
    force: bool = False

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)
        self.prefix = self.prefix.strip().strip("/")


def download_labels(config: LabelDownloadConfig) -> Path:
    """Download label annotation objects from S3 and write a merged JSONL file."""
    load_dotenv()

    if not config.prefix:
        raise RuntimeError("S3 prefix must be provided for label download.")

    output_path = config.output_path
    if output_path.exists() and not config.force:
        logger.info("Labels already exist at {}; skipping", output_path)
        return output_path

    settings = load_s3_settings(require_bucket=True)
    if settings.bucket is None:
        raise RuntimeError("AWS_BUCKET must be set.")
    bucket = settings.bucket
    s3_client = build_s3_client(settings)

    list_prefix = f"{config.prefix}/"
    keys = list(iter_keys(s3_client, bucket=bucket, prefix=list_prefix))
    if not keys:
        raise RuntimeError(f"No label files found at s3://{bucket}/{list_prefix}")

    logger.info(
        "Downloading {} label objects from s3://{}/{}",
        len(keys),
        bucket,
        list_prefix,
    )

    rows: list[dict] = []
    progress_step = max(1, len(keys) // 10)

    for idx, key in enumerate(keys, start=1):
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
        text = body.decode("utf-8")
        rows.append(json.loads(text))

        if idx == len(keys) or idx % progress_step == 0:
            pct = (idx / len(keys)) * 100
            logger.info("Label download progress: {}/{} ({:.1f}%)", idx, len(keys), pct)

    if not rows:
        raise RuntimeError(
            f"Found label files at s3://{bucket}/{list_prefix}, but parsed zero rows."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for obj in rows:
            handle.write(json.dumps(obj, ensure_ascii=False))
            handle.write("\n")

    logger.info("Wrote {} merged label rows to {}", len(rows), output_path)
    return output_path


__all__ = ["LabelDownloadConfig", "download_labels"]
