"""Upload generated pairwise labeling tasks to S3 under the `pairs/` prefix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from botocore.exceptions import ClientError
from dotenv import load_dotenv

from aitraf.logging import logger
from aitraf.utils.s3_utils import build_s3_client, load_s3_settings, object_exists


@dataclass
class PairUploadConfig:
    """Configuration for uploading generated pair files to S3."""

    pairs_dir: Path | str
    prefix: str = "pairs"
    force: bool = False

    def __post_init__(self) -> None:
        self.pairs_dir = Path(self.pairs_dir)
        self.prefix = self.prefix.strip().strip("/")


def upload_pairs(config: PairUploadConfig) -> int:
    """Upload pair files to `s3://$AWS_BUCKET/<prefix>/`."""
    load_dotenv()

    pairs_dir = config.pairs_dir
    if not pairs_dir.exists():
        raise RuntimeError(f"Pairs directory not found: {pairs_dir}")
    if not pairs_dir.is_dir():
        raise RuntimeError(f"Pairs path is not a directory: {pairs_dir}")
    if not config.prefix:
        raise RuntimeError("S3 prefix must be provided for pair upload.")

    settings = load_s3_settings(require_bucket=True)
    if settings.bucket is None:
        raise RuntimeError("AWS_BUCKET must be set.")
    bucket = settings.bucket

    files = sorted(p for p in pairs_dir.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"No pair files found under {pairs_dir}")

    s3_client = build_s3_client(settings)

    uploaded = 0
    skipped = 0
    failed = 0

    logger.info(
        "Uploading {} pair files to s3://{}/{}/ (force={})",
        len(files),
        bucket,
        config.prefix,
        config.force,
    )

    progress_step = max(1, len(files) // 10)

    for idx, path in enumerate(files, start=1):
        rel = path.relative_to(pairs_dir)
        key = f"{config.prefix}/{rel.as_posix()}"

        if not config.force and object_exists(s3_client, bucket=bucket, key=key):
            skipped += 1
            continue

        try:
            s3_client.upload_file(str(path), bucket, key)
        except ClientError as exc:  # pragma: no cover - log only
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
