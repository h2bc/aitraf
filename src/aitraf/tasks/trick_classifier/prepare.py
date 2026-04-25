"""Task-owned data preparation for trick classification."""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from aitraf.prepare import (
    build_clip_manifest_base,
    ensure_manifest_targets_clear,
    load_labels_df,
    require_minimum_rows,
    split_manifest_rows,
    write_manifest_splits,
    write_task_vocab,
)
from aitraf.data_ops.utils import apply_dtypes, validate_required_columns


VIDEO_COL = "video"
REQUIRED_LABEL_COLUMNS = ("video", "trick")
STRATIFY_COL = "trick"
STRATIFY_STRATEGY = "label"
VOCAB_COLUMNS = ("trick", "key_foot", "person")
MANIFEST_COLUMNS = (
    "video_id",
    "s3_path",
    "trick",
    "key_foot",
    "person",
    "execution_score",
)
MANIFEST_DTYPES = {
    "video_id": "string",
    "s3_path": "string",
    "trick": "string",
    "key_foot": "string",
    "person": "string",
    "execution_score": "string",
}


def run_prepare(task_cfg: DictConfig, prepare_cfg: DictConfig) -> None:
    """Prepare manifests and vocab for the trick-classification task."""

    labels_df = load_labels_df(task_cfg.labels_path)
    validate_required_columns(labels_df, *REQUIRED_LABEL_COLUMNS)

    filtered_df = labels_df.dropna(subset=list(REQUIRED_LABEL_COLUMNS)).reset_index(
        drop=True
    )
    require_minimum_rows(
        filtered_df,
        task_name=task_cfg.name,
        required_cols=REQUIRED_LABEL_COLUMNS,
    )

    manifest_df = _build_manifest_df(filtered_df)

    output_dir = ensure_manifest_targets_clear(
        task_cfg.manifests_dir,
        force=bool(prepare_cfg.force),
        vocab_path=task_cfg.get("vocab_path"),
    )

    write_task_vocab(
        manifest_df,
        path=task_cfg.vocab_path,
        categorical_columns=VOCAB_COLUMNS,
        force=bool(prepare_cfg.force),
    )

    splits = split_manifest_rows(
        manifest_df,
        task_name=task_cfg.name,
        val_ratio=float(prepare_cfg.val_ratio),
        test_ratio=float(prepare_cfg.test_ratio),
        split_seed=int(prepare_cfg.split_seed),
        stratify_col=STRATIFY_COL,
        stratify_strategy=STRATIFY_STRATEGY,
    )
    write_manifest_splits(splits, task_name=task_cfg.name, output_dir=output_dir)


def _build_manifest_df(labels_df: pd.DataFrame) -> pd.DataFrame:
    manifest_df = build_clip_manifest_base(labels_df, video_col=VIDEO_COL)

    for col in (
        "trick",
        "key_foot",
        "person",
        "execution_score",
    ):
        if col in labels_df.columns:
            manifest_df[col] = labels_df[col]

    manifest_df = manifest_df[
        [col for col in MANIFEST_COLUMNS if col in manifest_df.columns]
    ]
    return apply_dtypes(manifest_df, MANIFEST_DTYPES)
