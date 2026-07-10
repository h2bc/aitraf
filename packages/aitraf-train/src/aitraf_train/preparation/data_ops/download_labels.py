"""Download label annotations from S3 and merge into one JSONL file."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from aitraf_core.cache import with_file_cache
from aitraf_core.storage.s3 import (
    build_s3_client,
    iter_keys,
    load_s3_settings,
    read_s3_uri,
)
from aitraf_train.logging import logger
from aitraf_train.preparation.data_ops.schema import LabelsSchema
from aitraf_train.preparation.data_ops.utils import apply_dtypes, apply_processors


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
    result = with_file_cache(
        path=output_path,
        force=config.force,
        compute=lambda: _download_labels(config),
        on_cache_hit=lambda path: logger.info(
            "Labels already exist at {}; skipping", path
        ),
    )
    return output_path if result is None else result


def _download_labels(config: LabelDownloadConfig) -> Path:
    output_path = config.output_path
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

    rows: list[dict[str, Any]] = []
    progress_step = max(1, len(keys) // 10)

    for idx, key in enumerate(keys, start=1):
        body = read_s3_uri(f"s3://{bucket}/{key}", s3_client=s3_client)
        text = body.decode("utf-8")
        rows.append(_flatten_labels_annotation(json.loads(text)))

        if idx == len(keys) or idx % progress_step == 0:
            pct = (idx / len(keys)) * 100
            logger.info("Label download progress: {}/{} ({:.1f}%)", idx, len(keys), pct)

    if not rows:
        raise RuntimeError(
            f"Found label files at s3://{bucket}/{list_prefix}, but parsed zero rows."
        )

    df = pd.DataFrame(rows)

    df = df.pipe(apply_processors, processors=LabelsSchema.processors).pipe(
        apply_dtypes, dtypes=LabelsSchema.types
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    logger.info("Wrote {} merged label rows to {}", len(df), output_path)
    return output_path


def _flatten_labels_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    task = annotation["task"]
    completed_by = annotation["completed_by"]
    flat: dict[str, Any] = dict(task["data"])
    flat.update(
        annotation_id=annotation["id"],
        annotator=completed_by["id"],
        id=task["id"],
        created_at=annotation["created_at"],
        updated_at=annotation["updated_at"],
        lead_time=annotation["lead_time"],
    )

    for item in annotation["result"]:
        name = item["from_name"]
        value = item["value"]
        if name == "execution_score":
            flat["execution_score"] = value["rating"]
            continue
        payload = next(iter(value.values()))
        flat[name] = payload[0] if isinstance(payload, list) else payload

    return flat


__all__ = ["LabelDownloadConfig", "download_labels"]
