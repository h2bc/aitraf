"""Publish extracted pose artifacts to S3 for the demo serving surface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from aitraf_core.storage.s3 import build_s3_client, load_s3_settings

from aitraf_train.storage.artifacts import upload_directory


@dataclass
class PoseUploadConfig:
    """Configuration for uploading pose `.npz` files to S3."""

    poses_dir: Path | str
    prefix: str = "poses"
    force: bool = False

    def __post_init__(self) -> None:
        self.poses_dir = Path(self.poses_dir)
        self.prefix = self.prefix.strip().strip("/")


def upload_poses(config: PoseUploadConfig) -> int:
    """Upload pose artifacts to `s3://$AWS_BUCKET/<prefix>/`."""
    load_dotenv()

    settings = load_s3_settings(require_bucket=True)
    if settings.bucket is None:
        raise RuntimeError("AWS_BUCKET must be set.")

    return upload_directory(
        config.poses_dir,
        bucket=settings.bucket,
        prefix=config.prefix,
        force=config.force,
        s3_client=build_s3_client(settings),
    )


__all__ = ["PoseUploadConfig", "upload_poses"]
