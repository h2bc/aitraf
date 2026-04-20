"""Task-owned data preparation for pairwise score ranking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from aitraf.data_ops.task_preparation import (
    ensure_manifest_targets_clear,
    load_manifest_df,
    load_pairwise_labels_df,
    write_manifest_splits,
    write_task_vocab,
)
from aitraf.data_ops.utils import apply_dtypes, validate_required_columns
from aitraf.logging import logger


REQUIRED_LABEL_COLUMNS = ("left", "right", "pref")
VOCAB_COLUMNS = ("trick", "pair_label")
MANIFEST_COLUMNS = (
    "annotation_id",
    "task_id",
    "trick",
    "pair_label",
    "left_video_id",
    "right_video_id",
    "left_s3_path",
    "right_s3_path",
)
MANIFEST_DTYPES = {
    "annotation_id": "Int64",
    "task_id": "Int64",
    "trick": "string",
    "pair_label": "string",
    "left_video_id": "string",
    "right_video_id": "string",
    "left_s3_path": "string",
    "right_s3_path": "string",
}


def run_prepare(task_cfg: DictConfig, prepare_cfg: DictConfig) -> None:
    """Prepare split-safe pairwise manifests from a source task's manifests."""

    pairwise_labels_df = load_pairwise_labels_df(task_cfg.pairwise_labels_path)
    validate_required_columns(pairwise_labels_df, *REQUIRED_LABEL_COLUMNS)

    output_dir = ensure_manifest_targets_clear(
        task_cfg.manifests_dir,
        force=bool(prepare_cfg.force),
        vocab_path=task_cfg.get("vocab_path"),
    )

    source_dir = Path(task_cfg.source_manifests_dir)
    split_frames: dict[str, pd.DataFrame] = {}
    combined_frames: list[pd.DataFrame] = []

    for split_name in ("train", "val", "test"):
        source_manifest_df = load_manifest_df(source_dir / f"{split_name}.jsonl")
        split_frame = _build_pairwise_split_manifest_df(
            pairwise_labels_df,
            source_manifest_df,
            split_name=split_name,
            task_name=task_cfg.name,
        )
        split_frames[split_name] = split_frame
        combined_frames.append(split_frame)

    combined_df = pd.concat(combined_frames, ignore_index=True)
    if len(combined_df) < 3:
        raise RuntimeError(
            f"Need at least 3 labeled pairs across all splits for task '{task_cfg.name}'."
        )

    write_task_vocab(
        combined_df,
        path=task_cfg.vocab_path,
        categorical_columns=VOCAB_COLUMNS,
        force=bool(prepare_cfg.force),
    )
    write_manifest_splits(split_frames, task_name=task_cfg.name, output_dir=output_dir)


def _build_pairwise_split_manifest_df(
    pairwise_labels_df: pd.DataFrame,
    source_manifest_df: pd.DataFrame,
    *,
    split_name: str,
    task_name: str,
) -> pd.DataFrame:
    validate_required_columns(source_manifest_df, "s3_path", "video_id", "trick")

    source_manifest_df = source_manifest_df.dropna(
        subset=["s3_path", "video_id", "trick"]
    ).reset_index(drop=True)

    duplicate_paths = source_manifest_df["s3_path"].duplicated()
    if duplicate_paths.any():
        duplicates = sorted(
            source_manifest_df.loc[duplicate_paths, "s3_path"].astype(str).unique()
        )
        preview = ", ".join(duplicates[:5])
        raise RuntimeError(
            f"Source manifest for split '{split_name}' contains duplicate clip rows. "
            f"Examples: {preview}"
        )

    labels_df = pairwise_labels_df.copy()
    labels_df["pair_label"] = labels_df["pref"].apply(_extract_pair_label)

    missing_pair_label = labels_df["pair_label"].isna()
    if missing_pair_label.any():
        logger.warning(
            "Dropping {} pairwise rows without a selected preference before '{}'",
            int(missing_pair_label.sum()),
            split_name,
        )
        labels_df = labels_df.loc[~missing_pair_label].reset_index(drop=True)

    invalid_labels = ~labels_df["pair_label"].isin({"left", "right"})
    if invalid_labels.any():
        bad_rows = labels_df.loc[invalid_labels, ["annotation_id", "pair_label"]]
        raise RuntimeError(
            "Unsupported pair labels found in pairwise labels input: "
            f"{bad_rows.head(5).to_dict(orient='records')}"
        )

    clip_metadata = source_manifest_df[["s3_path", "video_id", "trick"]].copy()

    left_metadata = clip_metadata.rename(
        columns={
            "s3_path": "left_s3_path",
            "video_id": "left_video_id",
            "trick": "left_trick",
        }
    )
    right_metadata = clip_metadata.rename(
        columns={
            "s3_path": "right_s3_path",
            "video_id": "right_video_id",
            "trick": "right_trick",
        }
    )

    merged = labels_df.merge(
        left_metadata,
        how="inner",
        left_on="left",
        right_on="left_s3_path",
    ).merge(
        right_metadata,
        how="inner",
        left_on="right",
        right_on="right_s3_path",
    )

    _validate_pairwise_rows(merged, split_name=split_name, task_name=task_name)

    manifest_df = pd.DataFrame(
        {
            "annotation_id": merged.get("annotation_id"),
            "task_id": merged.get("task_id"),
            "trick": merged["left_trick"],
            "pair_label": merged["pair_label"],
            "left_video_id": merged["left_video_id"],
            "right_video_id": merged["right_video_id"],
            "left_s3_path": merged["left_s3_path"],
            "right_s3_path": merged["right_s3_path"],
        }
    )

    manifest_df = manifest_df[
        [col for col in MANIFEST_COLUMNS if col in manifest_df.columns]
    ]
    manifest_df = apply_dtypes(manifest_df, MANIFEST_DTYPES)

    if len(manifest_df) == 0:
        raise RuntimeError(
            f"No labeled pairs matched the source manifest split '{split_name}' for "
            f"task '{task_name}'."
        )

    return manifest_df


def _extract_pair_label(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("selected")
    return value


def _validate_pairwise_rows(
    merged: pd.DataFrame,
    *,
    split_name: str,
    task_name: str,
) -> None:
    mismatched = merged["left_trick"] != merged["right_trick"]
    if mismatched.any():
        examples = merged.loc[
            mismatched, ["left", "right", "left_trick", "right_trick"]
        ]
        raise RuntimeError(
            f"Pairwise rows for task '{task_name}' join clips with different tricks "
            f"in split '{split_name}': {examples.head(5).to_dict(orient='records')}"
        )

    if "trick" in merged.columns:
        task_trick = merged["trick"].notna()
        mismatched_task_trick = task_trick & (merged["trick"] != merged["left_trick"])
        if mismatched_task_trick.any():
            examples = merged.loc[
                mismatched_task_trick, ["left", "right", "trick", "left_trick"]
            ]
            raise RuntimeError(
                f"Pairwise labels disagree with source manifests in split '{split_name}' "
                f"for task '{task_name}': {examples.head(5).to_dict(orient='records')}"
            )
