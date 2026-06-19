"""Task-owned data preparation for pairwise score prediction."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from aitraf_train.prepare import (
    ensure_manifest_targets_clear,
    load_manifest_df,
    load_pairwise_labels_df,
    write_manifest_splits,
    write_task_vocab,
)
from aitraf_train.data_ops.utils import apply_dtypes, validate_required_columns
from aitraf_train.logging import logger


REQUIRED_LABEL_COLUMNS = ("left", "right", "pref", "split")
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
    source_manifest_df = pd.concat(
        [
            load_manifest_df(source_dir / f"{split_name}.jsonl")
            for split_name in ("train", "val", "test")
        ],
        ignore_index=True,
    )
    split_frames: dict[str, pd.DataFrame] = {}
    combined_frames: list[pd.DataFrame] = []

    for split_name in ("train", "val", "test"):
        split_frame = _build_pairwise_split_manifest_df(
            pairwise_labels_df,
            source_manifest_df,
            split_name=split_name,
            task_name=task_cfg.name,
            clips_dir=task_cfg.get("clips_dir"),
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
    clips_dir: Path | str | None = None,
) -> pd.DataFrame:
    validate_required_columns(source_manifest_df, "s3_path", "video_id", "trick")

    source_manifest_df = source_manifest_df.dropna(
        subset=["s3_path", "video_id", "trick"]
    ).reset_index(drop=True)
    source_manifest_df["match_s3_path"] = source_manifest_df["s3_path"].map(
        _canonicalize_clip_uri
    )
    source_manifest_df["s3_path"] = source_manifest_df["match_s3_path"]
    source_manifest_df["video_id"] = source_manifest_df.apply(
        lambda row: _resolve_local_video_id(
            row["video_id"],
            row["match_s3_path"],
            clips_dir=Path(clips_dir) if clips_dir else None,
        ),
        axis=1,
    )

    duplicate_paths = source_manifest_df["match_s3_path"].duplicated()
    if duplicate_paths.any():
        duplicate_count = int(duplicate_paths.sum())
        logger.warning(
            "Dropping {} duplicate source manifest clip rows after canonicalizing "
            "segment ids",
            duplicate_count,
        )
        source_manifest_df = source_manifest_df.drop_duplicates(
            subset=["match_s3_path"], keep="first"
        ).reset_index(drop=True)

    labels_df = pairwise_labels_df.loc[
        pairwise_labels_df["split"].astype(str) == split_name
    ].copy()
    labels_df["pair_label"] = labels_df["pref"].apply(_extract_pair_label)
    labels_df["left_match_s3_path"] = labels_df["left"].map(_canonicalize_clip_uri)
    labels_df["right_match_s3_path"] = labels_df["right"].map(_canonicalize_clip_uri)

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

    clip_metadata = source_manifest_df[
        ["match_s3_path", "s3_path", "video_id", "trick"]
    ].copy()

    left_metadata = clip_metadata.rename(
        columns={
            "match_s3_path": "left_match_s3_path",
            "s3_path": "left_s3_path",
            "video_id": "left_video_id",
            "trick": "left_trick",
        }
    )
    right_metadata = clip_metadata.rename(
        columns={
            "match_s3_path": "right_match_s3_path",
            "s3_path": "right_s3_path",
            "video_id": "right_video_id",
            "trick": "right_trick",
        }
    )

    merged = labels_df.merge(
        left_metadata,
        how="inner",
        on="left_match_s3_path",
    ).merge(
        right_metadata,
        how="inner",
        on="right_match_s3_path",
    )

    dropped_rows = len(labels_df) - len(merged)
    if dropped_rows > 0:
        logger.warning(
            "Dropping {} pairwise rows from split '{}' because one or both clips "
            "were not found in the source manifests",
            dropped_rows,
            split_name,
        )

    mismatched_tricks = merged["left_trick"] != merged["right_trick"]
    if mismatched_tricks.any():
        logger.warning(
            "Dropping {} pairwise rows from split '{}' because source manifests "
            "assign different tricks to the compared clips",
            int(mismatched_tricks.sum()),
            split_name,
        )
        merged = merged.loc[~mismatched_tricks].reset_index(drop=True)

    if "trick" in merged.columns:
        stale_task_trick = merged["trick"].notna() & (
            merged["trick"] != merged["left_trick"]
        )
        if stale_task_trick.any():
            logger.warning(
                "Dropping {} pairwise rows from split '{}' because pairwise labels "
                "disagree with source manifest tricks",
                int(stale_task_trick.sum()),
                split_name,
            )
            merged = merged.loc[~stale_task_trick].reset_index(drop=True)

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


def _canonicalize_clip_uri(value: Any) -> str:
    """Normalize old unpadded segment ids to the current clip naming convention."""

    return re.sub(
        r"-seg(\d+)(\.mp4)$",
        lambda match: f"-seg{int(match.group(1)):02d}{match.group(2)}",
        str(value),
    )


def _resolve_local_video_id(
    video_id: Any,
    canonical_s3_path: Any,
    *,
    clips_dir: Path | None,
) -> str:
    source_video_id = str(video_id)
    canonical_video_id = Path(str(canonical_s3_path)).name

    if clips_dir is None:
        return source_video_id
    if (clips_dir / source_video_id).exists():
        return source_video_id
    if (clips_dir / canonical_video_id).exists():
        return canonical_video_id
    return source_video_id


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
