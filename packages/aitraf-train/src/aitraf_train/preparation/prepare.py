"""Shared helpers for the prepare pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aitraf_train.preparation.data_ops import schema
from aitraf_train.preparation.data_ops.utils import (
    apply_dtypes,
    apply_processors,
    build_vocab_payload,
    get_stratify_labels,
    split_df,
    validate_required_columns,
    write_vocab_file,
)
from aitraf_train.logging import logger


def load_labels_df(path: Path | str) -> pd.DataFrame:
    """Read normalized pointwise labels from JSONL."""

    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Labels file not found: {path}")

    return (
        pd.read_json(path, orient="records", lines=True)
        .pipe(apply_processors, processors=schema.LabelsSchema.processors)
        .pipe(apply_dtypes, dtypes=schema.LabelsSchema.types)
    )


def load_pairwise_labels_df(path: Path | str) -> pd.DataFrame:
    """Read pairwise labels from JSONL."""

    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Pairwise labels file not found: {path}")

    return pd.read_json(path, orient="records", lines=True).pipe(
        apply_dtypes, dtypes=schema.PairwiseLabelsSchema.types
    )


def load_manifest_df(path: Path | str) -> pd.DataFrame:
    """Read a task manifest from JSONL."""

    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Manifest file not found: {path}")

    return pd.read_json(path, orient="records", lines=True)


def build_clip_manifest_base(
    df: pd.DataFrame,
    *,
    video_col: str,
) -> pd.DataFrame:
    """Build the shared clip-identification columns for a task manifest."""

    validate_required_columns(df, video_col)

    return pd.DataFrame(
        {
            "video_id": df[video_col].map(lambda value: Path(value).name),
            "s3_path": df[video_col],
        }
    )


def require_minimum_rows(
    df: pd.DataFrame,
    *,
    task_name: str,
    required_cols: tuple[str, ...],
    min_rows: int = 3,
) -> None:
    """Validate that a prepared task frame has enough rows to split."""

    if len(df) >= min_rows:
        return

    if required_cols:
        required_cols_csv = ", ".join(required_cols)
        raise RuntimeError(
            f"Need at least {min_rows} rows with required columns "
            f"({required_cols_csv}) for task '{task_name}'."
        )

    raise RuntimeError(f"Need at least {min_rows} rows for task '{task_name}'.")


def ensure_manifest_targets_clear(
    output_dir: Path | str,
    *,
    force: bool,
    vocab_path: Path | str | None = None,
) -> Path:
    """Fail early if a task would overwrite existing manifest files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = [output_dir / name for name in ("train.jsonl", "val.jsonl", "test.jsonl")]
    if vocab_path is not None:
        targets.append(Path(vocab_path))

    if force:
        return output_dir

    for path in targets:
        if path.exists():
            raise RuntimeError(
                f"{path} exists. Set force=true to overwrite task outputs."
            )

    return output_dir


def split_manifest_rows(
    manifest_df: pd.DataFrame,
    *,
    task_name: str,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    stratify_col: str | None,
    stratify_strategy: str,
) -> dict[str, pd.DataFrame]:
    """Split a task manifest into train/val/test frames."""

    val_ratio = float(val_ratio)
    test_ratio = float(test_ratio)
    train_ratio = 1.0 - (val_ratio + test_ratio)
    stratify_labels = get_stratify_labels(manifest_df, stratify_col, stratify_strategy)

    try:
        train_val_df, test_df = split_df(
            manifest_df,
            test_ratio,
            stratify_labels,
            seed=split_seed,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Failed to create the train/test split for task '{task_name}'."
        ) from exc

    val_fraction = val_ratio / (val_ratio + train_ratio)
    train_stratify_labels = (
        stratify_labels.loc[train_val_df.index] if stratify_labels is not None else None
    )

    try:
        train_df, val_df = split_df(
            train_val_df,
            val_fraction,
            train_stratify_labels,
            seed=split_seed,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Failed to create the train/val split for task '{task_name}'."
        ) from exc

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def write_manifest(df: pd.DataFrame, path: Path | str) -> None:
    """Write a JSONL manifest file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def write_manifest_splits(
    splits: dict[str, pd.DataFrame],
    *,
    task_name: str,
    output_dir: Path | str,
) -> None:
    """Persist split manifests for a task."""

    output_dir = Path(output_dir)
    for split_name, split_frame in splits.items():
        out_path = output_dir / f"{split_name}.jsonl"
        write_manifest(split_frame, out_path)
        logger.info(
            "Task '{}' wrote {} ({} rows)", task_name, out_path, len(split_frame)
        )


def write_task_vocab(
    manifest_df: pd.DataFrame,
    *,
    path: Path | str,
    categorical_columns: tuple[str, ...],
    force: bool,
) -> None:
    """Write a task-local vocabulary file."""

    vocab_payload = build_vocab_payload(manifest_df, categorical_columns)
    write_vocab_file(vocab_payload, Path(path).resolve(), force=force)
    logger.info("Wrote categorical vocab to {}", path)
