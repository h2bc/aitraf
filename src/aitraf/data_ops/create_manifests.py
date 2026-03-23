"""Split labels into train/val/test manifests per task."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from aitraf.data_ops import schema
from aitraf.data_ops.utils import (
    apply_dtypes,
    apply_processors,
    build_vocab_payload,
    get_stratify_labels,
    split_df,
    validate_required_columns,
    write_vocab_file,
)
from aitraf.logging import logger


@dataclass
class TaskConfig:
    """Task-specific manifest settings."""

    name: str
    video_col: str
    required_cols: Sequence[str]
    manifests_dir: Path | str | None = None
    vocab_path: Path | str | None = None
    stratify_col: str | None = None
    stratify_strategy: str = "label"

    def __post_init__(self) -> None:
        self.required_cols = tuple(self.required_cols)
        if self.manifests_dir is not None:
            self.manifests_dir = Path(self.manifests_dir)
        if self.vocab_path is not None:
            self.vocab_path = Path(self.vocab_path)


@dataclass
class ManifestBuildConfig:
    """Configuration for manifest generation."""

    input_path: Path | str
    output_dir: Path | str
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    force: bool = False
    tasks: Sequence[TaskConfig] | None = None

    def __post_init__(self) -> None:
        self.input_path = Path(self.input_path)
        self.output_dir = Path(self.output_dir)
        if self.tasks is None:
            self.tasks = ()
        else:
            self.tasks = tuple(self.tasks)


def create_manifests(config: ManifestBuildConfig) -> None:
    if not config.input_path.exists():
        raise RuntimeError(f"Input file not found: {config.input_path}")

    if not config.tasks:
        raise RuntimeError("No tasks provided for manifest creation.")

    labels_df = pd.read_json(config.input_path, orient="records", lines=True)
    labels_df = labels_df.pipe(apply_dtypes, dtypes=schema.LabelsSchema.types)

    for task in config.tasks:
        _build_task_manifests(labels_df, config, task)


def _build_task_manifests(
    labels_df: pd.DataFrame, config: ManifestBuildConfig, task: TaskConfig
) -> None:
    required_cols = tuple(task.required_cols)
    if required_cols:
        validate_required_columns(labels_df, *required_cols)

    task_output_dir = task.manifests_dir or config.output_dir / task.name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    if not config.force:
        for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
            out_path = task_output_dir / name
            if out_path.exists():
                raise RuntimeError(
                    f"{out_path} exists. Set force=true to overwrite manifests."
                )

    if required_cols:
        filtered_df = labels_df.dropna(subset=list(required_cols)).reset_index(
            drop=True
        )
    else:
        filtered_df = labels_df.reset_index(drop=True)

    if len(filtered_df) < 3:
        if required_cols:
            required_cols_csv = ", ".join(required_cols)
            raise RuntimeError(
                f"Need at least 3 rows with required columns ({required_cols_csv}) "
                f"for task '{task.name}'."
            )
        raise RuntimeError(f"Need at least 3 rows for task '{task.name}'.")

    manifest_df = _build_manifest_df(filtered_df, task.video_col)
    vocab_path = task.vocab_path or task_output_dir / "vocab.json"
    _write_vocab(
        manifest_df,
        vocab_path,
        categorical_columns=schema.ManifestsSchema.categorical,
        force=config.force,
    )

    val_ratio = float(config.val_ratio)
    test_ratio = float(config.test_ratio)
    train_ratio = 1.0 - (val_ratio + test_ratio)
    stratify_labels = get_stratify_labels(
        manifest_df, task.stratify_col, task.stratify_strategy
    )

    train_val_df, test_df = split_df(
        manifest_df,
        test_ratio,
        stratify_labels,
    )

    val_fraction = val_ratio / (val_ratio + train_ratio)

    train_stratify_labels = (
        stratify_labels.loc[train_val_df.index] if stratify_labels is not None else None
    )

    train_df, val_df = split_df(
        train_val_df,
        val_fraction,
        train_stratify_labels,
    )

    splits = {"train": train_df, "val": val_df, "test": test_df}

    for name, split_frame in splits.items():
        out_path = task_output_dir / f"{name}.jsonl"
        _write_manifest(split_frame, out_path)
        logger.info(
            "Task '{}' wrote {} ({} rows)", task.name, out_path, len(split_frame)
        )
def _build_manifest_df(df: pd.DataFrame, video_col: str) -> pd.DataFrame:
    sources = df[video_col]
    manifest_df = df[
        [col for col in schema.ManifestsSchema.columns if col in df.columns]
    ].copy()
    manifest_df["video_id"] = sources.map(lambda value: Path(value).name)
    manifest_df["s3_path"] = sources

    return manifest_df.pipe(
        apply_processors, processors=schema.ManifestsSchema.processors
    ).pipe(apply_dtypes, dtypes=schema.ManifestsSchema.types)


def _write_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def _write_vocab(
    manifest_df: pd.DataFrame,
    path: Path,
    *,
    categorical_columns: tuple[str, ...],
    force: bool,
) -> None:
    vocab_payload = build_vocab_payload(manifest_df, categorical_columns)
    write_vocab_file(vocab_payload, path.resolve(), force=force)
    logger.info("Wrote categorical vocab to {}", path)
