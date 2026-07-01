"""Download pairwise label annotation files from S3 and merge into one JSONL."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from aitraf_core.cache import with_file_cache
from aitraf_core.storage.s3 import build_s3_client, iter_keys, load_s3_settings
from aitraf_train.logging import logger
from aitraf_train.preparation.data_ops.schema import PairwiseLabelsSchema
from aitraf_train.preparation.data_ops.utils import apply_dtypes


@dataclass
class PairwiseLabelDownloadConfig:
    """Configuration for downloading and merging pairwise labels from S3."""

    prefix: str
    output_dir: Path | str
    output_path: Path | str
    force: bool = False

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_path = Path(self.output_path)
        self.prefix = self.prefix.strip().strip("/")


def download_pairwise_labels(config: PairwiseLabelDownloadConfig) -> Path:
    """Download pairwise label objects from S3 and write a merged JSONL file."""
    load_dotenv()

    if not config.prefix:
        raise RuntimeError("S3 prefix must be provided for pairwise label download.")

    settings = load_s3_settings(require_bucket=True)
    if settings.bucket is None:
        raise RuntimeError("AWS_BUCKET must be set.")
    bucket = settings.bucket
    s3_client = build_s3_client(settings)

    list_prefix = f"{config.prefix}/"
    keys = list(iter_keys(s3_client, bucket=bucket, prefix=list_prefix))
    if not keys:
        raise RuntimeError(
            f"No pairwise label files found at s3://{bucket}/{list_prefix}"
        )

    logger.info(
        "Downloading {} pairwise label objects from s3://{}/{}",
        len(keys),
        bucket,
        list_prefix,
    )

    output_dir = config.output_dir
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    progress_step = max(1, len(keys) // 10)

    for idx, key in enumerate(keys, start=1):
        relative_key = key[len(list_prefix) :] if key.startswith(list_prefix) else key
        local_path = output_dir / relative_key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with_file_cache(
            path=local_path,
            force=config.force,
            compute=lambda: s3_client.download_file(bucket, key, str(local_path)),
            on_cache_hit=lambda _: counts.update(skipped=1),
            on_cache_write=lambda _: counts.update(downloaded=1),
        )

        text = local_path.read_text(encoding="utf-8")
        records = _parse_payload(text)

        for record in records:
            rows.extend(_flatten_pairwise_label_records(record, source_key=key))

        if idx == len(keys) or idx % progress_step == 0:
            pct = (idx / len(keys)) * 100
            logger.info(
                "Pairwise label download progress: {}/{} ({:.1f}%)",
                idx,
                len(keys),
                pct,
            )

    if not rows:
        raise RuntimeError(
            f"Found pairwise label files at s3://{bucket}/{list_prefix}, but parsed zero rows."
        )

    output_path = config.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).pipe(apply_dtypes, dtypes=PairwiseLabelsSchema.types)
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    logger.info(
        "Pairwise label download summary: {} downloaded, {} skipped, {} merged rows written to {}",
        counts["downloaded"],
        counts["skipped"],
        len(df),
        output_path,
    )
    return output_path


def _flatten_pairwise_label_records(
    record: dict[str, Any], *, source_key: str
) -> list[dict[str, Any]]:
    """Convert one payload object into one or more flattened annotation rows."""
    if "result" in record:
        return [_flatten_pairwise_labels_annotation(record, source_key=source_key)]

    annotations = record.get("annotations")
    if isinstance(annotations, list):
        task_stub = {
            "id": record.get("id"),
            "data": record.get("data", {}),
        }
        flattened: list[dict[str, Any]] = []
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            ann_record = dict(ann)
            ann_record.setdefault("task", task_stub)
            flattened.append(
                _flatten_pairwise_labels_annotation(
                    ann_record,
                    source_key=source_key,
                )
            )
        return flattened

    return []


def _flatten_pairwise_labels_annotation(
    annotation: dict[str, Any], *, source_key: str
) -> dict[str, Any]:
    task = annotation.get("task") or {}
    task_data = task.get("data") or {}
    completed_by = annotation.get("completed_by") or {}

    row: dict[str, Any] = {
        "annotation_id": annotation.get("id"),
        "task_id": task.get("id"),
        "created_at": annotation.get("created_at"),
        "updated_at": annotation.get("updated_at"),
        "lead_time": annotation.get("lead_time"),
        "annotator_id": completed_by.get("id"),
        "annotator_email": completed_by.get("email"),
        "created_username": annotation.get("created_username"),
        "source_s3_key": source_key,
    }

    if isinstance(task_data, dict):
        for key, value in task_data.items():
            row[str(key)] = value

    result = annotation.get("result")
    if isinstance(result, list):
        for item in result:
            if not isinstance(item, dict):
                continue
            field_name = item.get("from_name")
            if not field_name:
                continue
            payload = item.get("value") or {}
            if not isinstance(payload, dict) or "selected" not in payload:
                raise RuntimeError(f"Unsupported pairwise result payload: {payload!r}")
            row[str(field_name)] = payload.get("selected")

    return row


def _parse_payload(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        for line in stripped.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        return rows

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [obj for obj in parsed if isinstance(obj, dict)]
    return []


__all__ = ["PairwiseLabelDownloadConfig", "download_pairwise_labels"]
