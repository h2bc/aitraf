"""Utilities for downloading referenced clip videos from S3."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from aitraf_core.cache import with_file_cache
from aitraf_train.logging import logger
from aitraf_train.preparation.data_ops import schema
from aitraf_train.preparation.data_ops.utils import strip_clips_prefix
from aitraf_train.utils.s3_utils import build_s3_client, load_s3_settings, parse_s3_uri


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
        schema.LabelsSchema.video_col
    ].astype(str)
    total_clips = len(clip_uris)

    output_dir = config.output_dir
    settings = load_s3_settings(require_bucket=False)
    s3_client = build_s3_client(settings)
    logger.info("Downloading {} clips into {}", total_clips, output_dir)
    counts: Counter[str] = Counter()

    progress_step = max(1, total_clips // 10)

    for idx, uri in enumerate(clip_uris, start=1):
        bucket, key = parse_s3_uri(uri)
        relative_key = strip_clips_prefix(Path(key))
        destination = output_dir / relative_key
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with_file_cache(
                path=destination,
                force=config.force,
                compute=lambda: s3_client.download_file(bucket, key, str(destination)),
                on_cache_hit=lambda _: counts.update(skipped=1),
                on_cache_write=lambda _: counts.update(downloaded=1),
            )
        except ClientError as exc:
            counts.update(failed=1)
            logger.warning("Failed to download {}: {}", uri, exc)

        if idx == total_clips or idx % progress_step == 0:
            pct = (idx / total_clips) * 100
            logger.info(
                "Clip download progress: {}/{} ({:.1f}%)", idx, total_clips, pct
            )

    logger.info(
        "Clip download summary: {} downloaded, {} skipped, {} failed (total {})",
        counts["downloaded"],
        counts["skipped"],
        counts["failed"],
        total_clips,
    )
