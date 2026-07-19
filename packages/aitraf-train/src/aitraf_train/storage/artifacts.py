"""Upload generated artifact directories to S3 under a fixed prefix."""

from __future__ import annotations

from pathlib import Path

from aitraf_core.storage.s3 import object_exists, upload_s3_uri

from aitraf_train.logging import logger


def upload_directory(
    directory: Path,
    *,
    bucket: str,
    prefix: str,
    force: bool,
    s3_client,
) -> int:
    """Upload every file under `directory` to `s3://bucket/prefix/`.

    Existing keys are skipped unless `force`. Any upload failure raises so a
    partially published artifact set is never mistaken for a complete one.
    """
    files = _collect_files(directory, prefix=prefix)

    logger.info(
        "Uploading {} files from {} to s3://{}/{}/ (force={})",
        len(files),
        directory,
        bucket,
        prefix,
        force,
    )

    uploaded = 0
    skipped = 0
    progress_step = max(1, len(files) // 10)

    for idx, path in enumerate(files, start=1):
        key = f"{prefix}/{path.relative_to(directory).as_posix()}"

        if not force and object_exists(s3_client, bucket=bucket, key=key):
            skipped += 1
            continue

        try:
            upload_s3_uri(path, f"s3://{bucket}/{key}", s3_client=s3_client)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to upload {path} to s3://{bucket}/{key}"
            ) from exc

        uploaded += 1
        if idx == len(files) or idx % progress_step == 0:
            pct = (idx / len(files)) * 100
            logger.info("Upload progress: {}/{} ({:.1f}%)", idx, len(files), pct)

    logger.info(
        "Upload summary: {} uploaded, {} skipped (total {})",
        uploaded,
        skipped,
        len(files),
    )
    return uploaded


def _collect_files(directory: Path, *, prefix: str) -> list[Path]:
    if not prefix:
        raise RuntimeError("S3 prefix must be provided for artifact upload.")
    if not directory.exists():
        raise RuntimeError(f"Artifact directory not found: {directory}")
    if not directory.is_dir():
        raise RuntimeError(f"Artifact path is not a directory: {directory}")

    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"No files found under {directory}")
    return files


__all__ = ["upload_directory"]
