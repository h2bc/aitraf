"""Label ops utilities for generating pairwise ranking tasks."""

from dataclasses import dataclass
from pathlib import Path
import json
import shutil
from itertools import combinations

import pandas as pd

from aitraf.data_ops import schema
from aitraf.data_ops.utils import apply_dtypes, validate_required_columns
from aitraf.logging import logger


@dataclass
class PairGenerationConfig:
    """Configuration for generating pairwise comparison tasks."""

    labels_path: Path | str
    output_dir: Path | str
    force: bool = False

    def __post_init__(self) -> None:
        self.labels_path = Path(self.labels_path)
        self.output_dir = Path(self.output_dir)


def create_pairs(config: PairGenerationConfig) -> int:
    """Create unique same-trick pairs and write one JSON file per pair."""
    labels_path = config.labels_path

    if not labels_path.exists():
        raise RuntimeError(f"Labels file not found: {labels_path}")

    labels_df = pd.read_json(labels_path, orient="records", lines=True)

    validate_required_columns(labels_df, "trick", "video")
    
    labels_df = (
        labels_df.pipe(apply_dtypes, dtypes=schema.LabelsSchema.types)
        .dropna(subset=["trick", "video"])
        .reset_index(drop=True)
    )

    output_dir = config.output_dir
    _prepare_output_dir(output_dir, force=config.force)

    total_pairs = 0
    skipped_tricks = 0

    for trick, group_df in labels_df.groupby("trick"):
        videos = group_df["video"].tolist()
        unique_videos = sorted(set(videos))

        if len(unique_videos) < 2:
            skipped_tricks += 1
            continue

        trick_name = str(trick)
        for idx, (left, right) in enumerate(combinations(unique_videos, 2), start=1):
            payload = {"data": {"trick": trick_name, "left": left, "right": right}}
            filename = f"{trick_name}__{idx:06d}.json"
            out_path = output_dir / filename
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            total_pairs += 1

    logger.info(
        "Created {} pair files in {} ({} tricks skipped)",
        total_pairs,
        output_dir,
        skipped_tricks,
    )

    if total_pairs == 0:
        raise RuntimeError("No pairs were created. Check label data for duplicates.")

    return total_pairs


def _prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if force:
            shutil.rmtree(output_dir)
        else:
            if any(output_dir.iterdir()):
                raise RuntimeError(
                    f"Output directory {output_dir} is not empty. Set force=true to overwrite."
                )
    output_dir.mkdir(parents=True, exist_ok=True)
