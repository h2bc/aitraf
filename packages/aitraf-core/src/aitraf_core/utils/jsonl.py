"""JSONL file helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl_records(path: Path | str) -> list[dict[str, Any]]:
    """Read a JSONL file containing one object per non-empty line."""

    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    if not jsonl_path.is_file():
        raise IsADirectoryError(f"JSONL path is not a file: {jsonl_path}")

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {jsonl_path}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected object at line {line_number}: {jsonl_path}"
                )
            rows.append(row)
    return rows


__all__ = ["read_jsonl_records"]
