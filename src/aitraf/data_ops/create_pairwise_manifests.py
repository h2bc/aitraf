"""Split pairwise label annotations into train/val/test manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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
class PairwiseTaskConfig:
    """Task-specific manifest settings for pairwise samples."""

    name: str
    pairwise_labels_path: Path | str
    labels_path: Path | str
    pairwise_labels_required_cols: Sequence[str]
    labels_required_cols: Sequence[str]
    manifests_dir: Path | str | None = None
    vocab_path: Path | str | None = None
    stratify_col: str | None = None
    stratify_strategy: str = "label"

    def __post_init__(self) -> None:
        self.pairwise_labels_path = Path(self.pairwise_labels_path)
        self.labels_path = Path(self.labels_path)
        self.pairwise_labels_required_cols = tuple(self.pairwise_labels_required_cols)
        self.labels_required_cols = tuple(self.labels_required_cols)
        if self.manifests_dir is not None:
            self.manifests_dir = Path(self.manifests_dir)
        if self.vocab_path is not None:
            self.vocab_path = Path(self.vocab_path)


@dataclass
class PairwiseManifestBuildConfig:
    """Configuration for pairwise manifest generation."""

    output_dir: Path | str
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    force: bool = False
    tasks: Sequence[PairwiseTaskConfig] | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.tasks is None:
            self.tasks = ()
        else:
            self.tasks = tuple(self.tasks)


def create_pairwise_manifests(config: PairwiseManifestBuildConfig) -> None:
    if not config.tasks:
        raise RuntimeError("No tasks provided for pairwise manifest creation.")

    for task in config.tasks:
        _build_task_manifests(config, task)


def _build_task_manifests(
    config: PairwiseManifestBuildConfig, task: PairwiseTaskConfig
) -> None:
    if not task.pairwise_labels_path.exists():
        raise RuntimeError(
            f"Pairwise labels file not found: {task.pairwise_labels_path}"
        )
    if not task.labels_path.exists():
        raise RuntimeError(f"Labels file not found: {task.labels_path}")

    task_output_dir = task.manifests_dir or config.output_dir / task.name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    if not config.force:
        for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
            out_path = task_output_dir / name
            if out_path.exists():
                raise RuntimeError(
                    f"{out_path} exists. Set force=true to overwrite manifests."
                )

    pairwise_labels_df = pd.read_json(
        task.pairwise_labels_path, orient="records", lines=True
    )
    labels_df = pd.read_json(task.labels_path, orient="records", lines=True)

    manifest_df = _build_pairwise_manifest_df(
        pairwise_labels_df,
        labels_df,
        pairwise_labels_required_cols=task.pairwise_labels_required_cols,
        labels_required_cols=task.labels_required_cols,
    )

    if len(manifest_df) < 3:
        raise RuntimeError(f"Need at least 3 rows for pairwise task '{task.name}'.")

    vocab_path = task.vocab_path or task_output_dir / "vocab.json"
    _write_vocab(
        manifest_df,
        vocab_path,
        categorical_columns=schema.PairwiseManifestsSchema.categorical,
        force=config.force,
    )

    val_ratio = float(config.val_ratio)
    test_ratio = float(config.test_ratio)
    train_ratio = 1.0 - (val_ratio + test_ratio)
    stratify_labels = get_stratify_labels(
        manifest_df, task.stratify_col, task.stratify_strategy
    )

    try:
        train_val_df, test_df = split_df(
            manifest_df,
            test_ratio,
            stratify_labels,
            seed=config.split_seed,
        )
    except ValueError as exc:
        raise RuntimeError(
            _build_split_error(task.name, "train/test", stratify_labels)
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
            seed=config.split_seed,
        )
    except ValueError as exc:
        raise RuntimeError(
            _build_split_error(task.name, "train/val", train_stratify_labels)
        ) from exc

    for split_name, split_frame in {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }.items():
        out_path = task_output_dir / f"{split_name}.jsonl"
        split_frame.to_json(out_path, orient="records", lines=True, force_ascii=False)
        logger.info(
            "Task '{}' wrote {} ({} rows)", task.name, out_path, len(split_frame)
        )


def _build_pairwise_manifest_df(
    pairwise_labels_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    pairwise_labels_required_cols: Sequence[str],
    labels_required_cols: Sequence[str],
) -> pd.DataFrame:
    validate_required_columns(pairwise_labels_df, *tuple(pairwise_labels_required_cols))
    validate_required_columns(labels_df, *tuple(labels_required_cols))

    pairwise_labels_df = pairwise_labels_df.pipe(
        apply_dtypes, dtypes=schema.PairwiseLabelsSchema.types
    )
    labels_df = labels_df.pipe(
        apply_processors, processors=schema.LabelsSchema.processors
    )
    labels_df = labels_df.pipe(apply_dtypes, dtypes=schema.LabelsSchema.types)
    labels_df = labels_df.dropna(subset=["video", "trick"]).reset_index(drop=True)

    duplicate_videos = labels_df["video"].duplicated()
    if duplicate_videos.any():
        duplicates = sorted(
            labels_df.loc[duplicate_videos, "video"].astype(str).unique()
        )
        preview = ", ".join(duplicates[:5])
        raise RuntimeError(
            "Labels contain duplicate video rows; cannot build pairwise manifests. "
            f"Examples: {preview}"
        )

    pairwise_labels_df = pairwise_labels_df.copy()
    pairwise_labels_df["pair_label"] = pairwise_labels_df["pref"].apply(
        _extract_pair_label
    )

    missing_pair_label = pairwise_labels_df["pair_label"].isna()
    if missing_pair_label.any():
        logger.warning(
            "Dropping {} pairwise label rows without a selected preference",
            int(missing_pair_label.sum()),
        )
        pairwise_labels_df = pairwise_labels_df.loc[~missing_pair_label].reset_index(
            drop=True
        )

    invalid_labels = ~pairwise_labels_df["pair_label"].isin({"left", "right"})
    if invalid_labels.any():
        bad_rows = pairwise_labels_df.loc[
            invalid_labels, ["annotation_id", "pair_label"]
        ]
        raise RuntimeError(
            "Unsupported pair labels found in pairwise labels input: "
            f"{bad_rows.head(5).to_dict(orient='records')}"
        )

    label_metadata = labels_df[["video", "trick"]].copy()

    left_metadata = label_metadata.rename(
        columns={
            "video": "left_s3_path",
            "trick": "left_trick",
        }
    )
    right_metadata = label_metadata.rename(
        columns={
            "video": "right_s3_path",
            "trick": "right_trick",
        }
    )

    merged = pairwise_labels_df.merge(
        left_metadata,
        how="left",
        left_on="left",
        right_on="left_s3_path",
    ).merge(
        right_metadata,
        how="left",
        left_on="right",
        right_on="right_s3_path",
    )

    _validate_pairwise_merge(merged)

    manifest_df = pd.DataFrame(
        {
            "annotation_id": merged.get("annotation_id"),
            "task_id": merged.get("task_id"),
            "trick": merged["left_trick"],
            "pair_label": merged["pair_label"],
            "left_video_id": merged["left"].map(lambda value: Path(value).name),
            "right_video_id": merged["right"].map(lambda value: Path(value).name),
            "left_s3_path": merged["left"],
            "right_s3_path": merged["right"],
        }
    )

    manifest_df = manifest_df[
        [
            col
            for col in schema.PairwiseManifestsSchema.columns
            if col in manifest_df.columns
        ]
    ]
    return manifest_df.pipe(apply_dtypes, dtypes=schema.PairwiseManifestsSchema.types)


def _extract_pair_label(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("selected")
    return value


def _validate_pairwise_merge(merged: pd.DataFrame) -> None:
    missing_left = merged["left_trick"].isna()
    if missing_left.any():
        examples = merged.loc[missing_left, "left"].astype(str).head(5).tolist()
        raise RuntimeError(
            "Some pairwise labels reference left clips missing from labels.jsonl: "
            + ", ".join(examples)
        )

    missing_right = merged["right_trick"].isna()
    if missing_right.any():
        examples = merged.loc[missing_right, "right"].astype(str).head(5).tolist()
        raise RuntimeError(
            "Some pairwise labels reference right clips missing from labels.jsonl: "
            + ", ".join(examples)
        )

    mismatched = merged["left_trick"] != merged["right_trick"]
    if mismatched.any():
        examples = merged.loc[
            mismatched, ["left", "right", "left_trick", "right_trick"]
        ]
        raise RuntimeError(
            "Some pairwise labels join clips with different label tricks: "
            f"{examples.head(5).to_dict(orient='records')}"
        )


def _build_split_error(
    task_name: str,
    split_name: str,
    stratify_labels: pd.Series | None,
) -> str:
    if stratify_labels is None:
        return (
            f"Failed to create the {split_name} split for pairwise task '{task_name}'."
        )

    counts = stratify_labels.value_counts().sort_index()
    counts_text = ", ".join(f"{label}={count}" for label, count in counts.items())
    return (
        f"Failed to create the {split_name} split for pairwise task '{task_name}' "
        f"with stratify labels: {counts_text}"
    )


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
