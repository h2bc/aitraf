#!/usr/bin/env python3
"""
Read the Label Studio parquet export, split it, and write JSONL manifests.

Usage:
    python -m scripts.build_manifests [--input data/label_studio_export.parquet]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from paths import DATA_DIR
from dataset_schema import TARGET_COLUMN, EXPECTED_COLUMNS, CATEGORICAL_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test manifests.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "labeled.parquet",
        help="Parquet file produced by scripts/pull_ls.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "manifests",
        help="Directory where JSONL files will be written.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=None,
        help="Optional path for label vocab metadata (defaults to output-dir/labels.json).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Portion reserved for validation (default 0.1).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Portion reserved for test (default 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splits.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing manifest/label files if they already exist.",
    )
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Input is missing expected columns: {', '.join(missing)}. "
            "Re-run scripts/pull_ls.py."
        )


def sanitized_split(
    data: pd.DataFrame,
    labels: pd.Series,
    test_size: float,
    seed: int,
    msg: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if test_size <= 0:
        return data, data.iloc[0:0], labels, labels.iloc[0:0]
    try:
        return train_test_split(
            data,
            labels,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
    except ValueError as exc:
        print(f"Warning: {msg} ({exc}). Using unstratified split.")
        return train_test_split(
            data,
            labels,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in df[EXPECTED_COLUMNS].to_dict(orient="records"):
            json.dump(record, fh, ensure_ascii=True)
            fh.write("\n")


def build_vocab_metadata(df: pd.DataFrame) -> dict[str, list[str]]:
    """Collect categorical vocabularies for the target and optional feature columns."""
    vocabs: dict[str, list[str]] = {}
    for col in CATEGORICAL_COLUMNS:
        series = df[col].dropna().astype(str)
        vocabs[col] = sorted(series.unique().tolist())
    return vocabs


def write_label_metadata(vocabs: dict[str, list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for col, labels in vocabs.items():
        payload[col] = {
            "labels": labels,
            "label2id": {label: idx for idx, label in enumerate(labels)},
            "id2label": {str(idx): label for idx, label in enumerate(labels)},
        }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)


def ensure_can_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(
            f"Output file {path} already exists. Use --force to overwrite."
        )


def summarize_distribution(
    df: pd.DataFrame, column: str, indent: str = " " * 10
) -> str:
    """Return counts and percentages for each class in a column (one per line)."""
    if df.empty:
        return f"{indent}(no rows)"
    counts = df[column].astype(str).value_counts().sort_index()
    total = counts.sum()
    label_width = max(len(label) for label in counts.index)
    count_width = max(len(str(count)) for count in counts.values)
    lines = []
    for label, count in counts.items():
        pct = count / total * 100
        lines.append(
            f"{indent}{label:<{label_width}}  {count:>{count_width}} ({pct:5.1f}%)"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Parquet input not found: {args.input}")

    if args.val_ratio < 0 or args.test_ratio < 0:
        raise SystemExit("Split ratios must be non-negative.")

    holdout = args.val_ratio + args.test_ratio
    train_ratio = 1.0 - holdout
    if train_ratio <= 0:
        raise SystemExit("Validation + test ratios must be < 1.")

    df = pd.read_parquet(args.input)
    ensure_columns(df)
    df = df.dropna(subset=EXPECTED_COLUMNS).reset_index(drop=True)
    if len(df) < 3:
        raise SystemExit("Need at least 3 fully labeled rows to perform splits.")

    stratify_labels = df[TARGET_COLUMN].astype(str)
    label_vocabs = build_vocab_metadata(df)

    train_val_df, test_df, train_val_labels, _ = sanitized_split(
        df,
        stratify_labels,
        args.test_ratio,
        args.seed,
        "test split failed",
    )

    val_fraction = args.val_ratio / (args.val_ratio + train_ratio)
    train_df, val_df, _, _ = sanitized_split(
        train_val_df,
        train_val_labels,
        val_fraction,
        args.seed + 1,
        "val split failed",
    )

    outputs = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    total_rows = len(df)

    for name, split_df in outputs.items():
        out_path = args.output_dir / f"{name}.jsonl"
        ensure_can_write(out_path, args.force)
        write_jsonl(split_df, out_path)
        pct = (len(split_df) / total_rows * 100) if total_rows else 0.0
        print(f"{name:>5}: {len(split_df):>5} rows ({pct:5.1f}%) -> {out_path}")
        print("      trick mix:")
        print(summarize_distribution(split_df, TARGET_COLUMN, indent=" " * 8))

    labels_path = args.labels_path or (args.output_dir / "labels.json")
    ensure_can_write(labels_path, args.force)
    write_label_metadata(label_vocabs, labels_path)
    print(f"labels: {labels_path}")


if __name__ == "__main__":
    main()
