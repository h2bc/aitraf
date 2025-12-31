"""Split Label Studio export into train/val/test manifests per task."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from aitraf.data_ops import schema
from aitraf.data_ops.utils import apply_dtypes, apply_processors
from aitraf.logging import logger


@dataclass
class TaskConfig:
    """Task-specific manifest settings."""

    name: str
    target_column: str
    manifests_dir: Path | str | None = None
    stratify_by_target: bool = True
    task_type: str | None = None

    def __post_init__(self) -> None:
        if self.manifests_dir is not None:
            self.manifests_dir = Path(self.manifests_dir)


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

    built_any = False
    for task in config.tasks:
        target_column = task.target_column

        if not _check_required_columns(labels_df, task.name, target_column):
            continue

        _build_task_manifests(labels_df, config, task)

        built_any = True

    if not built_any:
        raise RuntimeError(
            "No manifests were created because none of the requested target columns exist in the labels."
        )

    _write_vocab(labels_df, config.output_dir, force=config.force)


def _build_task_manifests(
    labels_df: pd.DataFrame, config: ManifestBuildConfig, task: TaskConfig
) -> None:
    target_column = task.target_column
    _validate_required_columns(labels_df, schema.LabelsSchema.input_col, target_column)

    task_output_dir = task.manifests_dir or config.output_dir / task.name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    if not config.force:
        for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
            out_path = task_output_dir / name
            if out_path.exists():
                raise RuntimeError(
                    f"{out_path} exists. Set force=true to overwrite manifests."
                )

    filtered_df = labels_df.dropna(
        subset=[schema.LabelsSchema.input_col, target_column]
    ).reset_index(drop=True)

    if len(filtered_df) < 3:
        raise RuntimeError(
            f"Need at least 3 rows with '{schema.LabelsSchema.input_col}' and '{target_column}' "
            f"for task '{task.name}'."
        )

    manifest_df = _build_manifest_df(filtered_df, target_column)

    val_ratio = float(config.val_ratio)
    test_ratio = float(config.test_ratio)
    train_ratio = 1.0 - (val_ratio + test_ratio)
    stratify_labels = _get_stratify_labels(
        manifest_df, target_column, task.stratify_by_target, task.task_type
    )

    train_val_df, test_df = _split(
        manifest_df,
        test_ratio,
        stratify_labels,
    )

    val_fraction = val_ratio / (val_ratio + train_ratio)

    train_stratify_labels = (
        stratify_labels.loc[train_val_df.index] if stratify_labels is not None else None
    )

    train_df, val_df = _split(
        train_val_df,
        val_fraction,
        train_stratify_labels,
    )

    splits = {"train": train_df, "val": val_df, "test": test_df}

    for name, split_df in splits.items():
        out_path = task_output_dir / f"{name}.jsonl"
        _write_manifest(split_df, out_path)
        logger.info("Task '{}' wrote {} ({} rows)", task.name, out_path, len(split_df))


def _validate_required_columns(df: pd.DataFrame, *columns: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Input is missing required columns: " + ", ".join(sorted(missing))
        )


def _check_required_columns(df: pd.DataFrame, task_name: str, *columns: str) -> bool:
    missing = [c for c in columns if c not in df.columns]

    if not missing:
        return True

    quoted = ", ".join(f"'{col}'" for col in missing)

    logger.warning(
        "Skipping task '{}': missing required columns {} in labels",
        task_name,
        quoted,
    )

    return False


def _split(
    df: pd.DataFrame, fraction: float, stratify: pd.Series | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < fraction < 1:
        raise RuntimeError("Split fraction must be between 0 and 1.")

    train_df, test_df = train_test_split(
        df,
        test_size=fraction,
        stratify=stratify,
    )
    return train_df, test_df


def _get_stratify_labels(
    df: pd.DataFrame, target_column: str, enabled: bool, task_type: str | None
) -> pd.Series | None:
    if not enabled:
        return None

    is_numerical = str(task_type).lower() == "regression" if task_type else False

    labels = df[target_column]
    if is_numerical:
        binned = pd.qcut(labels, q=min(5, len(labels)), duplicates="drop")
        if binned.nunique() < 2:
            raise RuntimeError(
                f"Cannot stratify '{target_column}': not enough variation to form bins."
            )
        return binned.astype(str)

    return labels.astype(str)


def _build_manifest_df(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    extra_columns = [
        col
        for col in schema.ManifestSchema.columns
        if col not in ("video_id", "s3_path", target_column) and col in df.columns
    ]

    sources = df[schema.LabelsSchema.input_col]

    manifest_df = pd.DataFrame(
        {
            "video_id": sources.map(lambda value: Path(value).name),
            "s3_path": sources,
            target_column: df[target_column],
        }
    )
    for col in extra_columns:
        manifest_df[col] = df[col]

    return manifest_df.pipe(
        apply_processors, processors=schema.ManifestSchema.processors
    ).pipe(apply_dtypes, dtypes=schema.ManifestSchema.types)


def _write_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def _build_vocab_metadata(
    df: pd.DataFrame,
) -> dict[str, dict[str, dict[str, str] | list[str]]]:
    payload = {}
    for col in schema.ManifestSchema.categorical:
        if col not in df.columns:
            continue
        labels = sorted(df[col].dropna().unique().tolist())
        payload[col] = {
            "labels": labels,
            "label2id": {label: idx for idx, label in enumerate(labels)},
            "id2label": {str(idx): label for idx, label in enumerate(labels)},
        }
    return payload


def _write_vocab(
    labels_df: pd.DataFrame,
    output_dir: Path | str,
    *,
    force: bool,
) -> None:
    vocab_payload = _build_vocab_metadata(labels_df)
    vocab_path = (Path(output_dir) / "vocab.json").resolve()
    if vocab_path.exists() and not force:
        raise RuntimeError(
            f"Vocabulary file already exists at {vocab_path}. Set force=true to overwrite."
        )

    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(
        json.dumps(vocab_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote categorical vocab to {}", vocab_path)
