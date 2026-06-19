"""Task-owned data preparation for ordinal score prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from aitraf_train.prepare import (
    build_clip_manifest_base,
    ensure_manifest_targets_clear,
    load_labels_df,
    require_minimum_rows,
    split_manifest_rows,
    write_manifest_splits,
)
from aitraf_train.data_ops.utils import (
    apply_dtypes,
    build_vocab_payload,
    validate_required_columns,
    write_vocab_file,
)


VIDEO_COL = "video"
REQUIRED_LABEL_COLUMNS = ("video", "execution_score")
STRATIFY_COL = "execution_score"
STRATIFY_STRATEGY = "label"
VOCAB_COLUMNS = ("execution_score", "trick", "key_foot", "person")
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
    """Prepare manifests for ordinal score prediction."""

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

    _write_vocab(
        manifest_df,
        path=task_cfg.vocab_path,
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


def _write_vocab(
    manifest_df: pd.DataFrame,
    *,
    path: Path | str,
    force: bool,
) -> None:
    vocab_payload = build_vocab_payload(
        manifest_df,
        tuple(col for col in VOCAB_COLUMNS if col != "execution_score"),
    )
    score_labels = sorted(
        manifest_df["execution_score"].dropna().unique().tolist(),
        key=lambda value: float(value),
    )
    vocab_payload = {
        "execution_score": {
            "labels": score_labels,
            "label2id": {label: idx for idx, label in enumerate(score_labels)},
            "id2label": {str(idx): label for idx, label in enumerate(score_labels)},
        },
        **vocab_payload,
    }
    write_vocab_file(vocab_payload, Path(path).resolve(), force=force)


__all__ = ["run_prepare"]
