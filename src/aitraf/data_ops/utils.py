"""Small helpers shared across data scripts."""

from pathlib import Path
from typing import Callable, Mapping
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def strip_clips_prefix(path: Path) -> Path:
    """Drop a leading 'clips/' prefix so downloads land under data/clips."""
    parts = path.parts
    if parts and parts[0] == "clips":
        parts = parts[1:]
    if not parts:
        return Path(path.name)
    return Path(*parts)


def apply_processors(
    df: pd.DataFrame,
    processors: Mapping[str, Callable[[object], object]],
) -> pd.DataFrame:
    for col, fn in processors.items():
        if col in df.columns:
            df[col] = df[col].apply(
                lambda value: value if pd.isna(value) else fn(value)
            )
    return df


def apply_dtypes(
    df: pd.DataFrame,
    dtypes: Mapping[str, str],
) -> pd.DataFrame:
    for col, dtype in dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df


def validate_required_columns(df: pd.DataFrame, *columns: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Input is missing required columns: " + ", ".join(sorted(missing))
        )


def split_df(
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


def get_stratify_labels(
    df: pd.DataFrame, stratify_col: str | None, strategy: str
) -> pd.Series | None:
    if stratify_col is None:
        return None

    labels = df[stratify_col]

    if strategy == "binned":
        binned = pd.qcut(labels, q=min(5, len(labels)), duplicates="drop")
        if binned.nunique() < 2:
            raise RuntimeError(
                f"Cannot stratify '{stratify_col}': not enough variation to form bins."
            )
        return binned.astype(str)

    if strategy == "label":
        return labels.astype(str)

    raise RuntimeError(
        f"Unsupported stratify_strategy '{strategy}'. Expected one of: label, binned"
    )


def build_vocab_payload(
    df: pd.DataFrame, categorical_columns: tuple[str, ...]
) -> dict[str, dict[str, dict[str, str] | list[str]]]:
    payload = {}
    for col in categorical_columns:
        if col not in df.columns:
            continue
        labels = sorted(df[col].dropna().unique().tolist())
        payload[col] = {
            "labels": labels,
            "label2id": {label: idx for idx, label in enumerate(labels)},
            "id2label": {str(idx): label for idx, label in enumerate(labels)},
        }
    return payload


def write_vocab_file(
    vocab_payload: dict[str, dict[str, dict[str, str] | list[str]]],
    path: Path,
    *,
    force: bool,
) -> None:
    if path.exists() and not force:
        raise RuntimeError(
            f"Vocabulary file already exists at {path}. Set force=true to overwrite."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(vocab_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
