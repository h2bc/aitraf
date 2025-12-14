"""Split Label Studio export into train/val/test manifests."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from aitraf.data_ops import schema
from aitraf.logging import logger


@dataclass
class ManifestBuildConfig:
    """Configuration for manifest generation."""

    input_path: Path | str
    output_dir: Path | str
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    force: bool = False

    def __post_init__(self) -> None:
        self.input_path = Path(self.input_path)
        self.output_dir = Path(self.output_dir)


def create_manifests(config: ManifestBuildConfig) -> None:
    input_path = config.input_path
    output_dir = config.output_dir
    if not input_path.exists():
        raise RuntimeError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not config.force:
        for name in ("train.jsonl", "val.jsonl", "test.jsonl", "labels.json"):
            out_path = output_dir / name
            if out_path.exists():
                raise RuntimeError(
                    f"Output file {out_path} already exists. Set force=true to overwrite."
                )

    logger.info("Loading labels from {}", input_path)
    df = pd.read_json(input_path, orient="records", lines=True)

    _ensure_columns(df)

    df = df.dropna(subset=schema.EXPECTED_COLUMNS).reset_index(drop=True)
    logger.info("Preparing manifests from {} labeled rows", len(df))

    if len(df) < 3:
        raise RuntimeError("Need at least 3 fully labeled rows to perform splits.")

    val_ratio = float(config.val_ratio)
    test_ratio = float(config.test_ratio)

    if val_ratio < 0 or test_ratio < 0:
        raise RuntimeError("Split ratios must be non-negative.")

    holdout = val_ratio + test_ratio
    train_ratio = 1.0 - holdout

    if train_ratio <= 0:
        raise RuntimeError("Validation + test ratios must be < 1.")

    stratify_labels = df[schema.TARGET_COLUMN].astype(str)
    train_val_df, test_df, train_val_labels, _ = _split(
        df,
        stratify_labels,
        test_ratio,
        int(config.seed),
    )

    val_fraction = val_ratio / (val_ratio + train_ratio)
    train_df, val_df, _, _ = _split(
        train_val_df,
        train_val_labels,
        val_fraction,
        int(config.seed) + 1,
    )

    outputs = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    for name, split_df in outputs.items():
        out_path = output_dir / f"{name}.jsonl"
        _write_manifest(split_df, out_path)
        logger.info("Wrote {} ({} rows)", out_path, len(split_df))

    labels_path = output_dir / "labels.json"
    labels_path.write_text(
        json.dumps(_build_vocab_metadata(df), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote vocab metadata to {}", labels_path)


def _ensure_columns(df: pd.DataFrame) -> None:
    missing = [c for c in schema.EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Input is missing expected columns: {', '.join(missing)}. Re-run the download step."
        )


def _split(df, labels, fraction, seed):
    if fraction <= 0:
        return df, df.iloc[0:0], labels, labels.iloc[0:0]

    try:
        return train_test_split(
            df,
            labels,
            test_size=fraction,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        return train_test_split(
            df,
            labels,
            test_size=fraction,
            random_state=seed,
            stratify=None,
        )


def _write_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            video_path = str(row[schema.VIDEO_COLUMN])
            record = {
                "video_id": Path(video_path).name,
                "s3_path": video_path,
                schema.TARGET_COLUMN: row[schema.TARGET_COLUMN],
            }
            for col in schema.CONTEXT_COLUMNS:
                record[col] = row[col]
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_vocab_metadata(
    df: pd.DataFrame,
) -> dict[str, dict[str, dict[str, str] | list[str]]]:
    payload = {}
    for col in schema.CATEGORICAL_COLUMNS:
        labels = sorted(df[col].dropna().astype(str).unique().tolist())
        payload[col] = {
            "labels": labels,
            "label2id": {label: idx for idx, label in enumerate(labels)},
            "id2label": {str(idx): label for idx, label in enumerate(labels)},
        }
    return payload
