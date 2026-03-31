"""Label ops utilities for generating pairwise ranking tasks."""

from dataclasses import dataclass
from pathlib import Path
import json
import random
import re
import shutil

import pandas as pd

from aitraf.data_ops import schema
from aitraf.data_ops.utils import apply_dtypes, validate_required_columns
from aitraf.logging import logger


@dataclass
class PairGenerationConfig:
    """Configuration for generating pairwise comparison tasks."""

    labels_path: Path | str
    output_dir: Path | str
    k_per_video: int
    seed: int = 42
    force: bool = False

    def __post_init__(self) -> None:
        self.labels_path = Path(self.labels_path)
        self.output_dir = Path(self.output_dir)


def create_pairs(config: PairGenerationConfig) -> int:
    """Create unique same-trick pairs and write one JSON file per pair."""
    labels_path = config.labels_path

    if not labels_path.exists():
        raise RuntimeError(f"Labels file not found: {labels_path}")
    if config.k_per_video < 0:
        raise ValueError("k_per_video must be >= 0.")

    labels_df = pd.read_json(labels_path, orient="records", lines=True)

    validate_required_columns(labels_df, "trick", "video")

    labels_df = (
        labels_df.pipe(apply_dtypes, dtypes=schema.LabelsSchema.types)
        .dropna(subset=["trick", "video"])
        .reset_index(drop=True)
    )

    output_dir = config.output_dir
    _prepare_output_dir(output_dir, force=config.force)
    existing_pairs_by_trick, existing_videos_by_trick, next_index_by_trick = (
        _load_existing_pairs(output_dir)
    )

    created_pairs = 0
    skipped_tricks = 0
    rng = random.Random(config.seed)

    for trick, group_df in labels_df.groupby("trick"):
        videos = group_df["video"].tolist()
        unique_videos = sorted(set(videos))
        trick_name = str(trick)

        if len(unique_videos) < 2:
            skipped_tricks += 1
            continue

        existing_pairs = existing_pairs_by_trick.get(trick_name, set())
        existing_videos = existing_videos_by_trick.get(trick_name, set())
        new_videos = sorted(set(unique_videos) - existing_videos)

        if not existing_pairs:
            pairs = _generate_pairs_for_all_videos(
                unique_videos=unique_videos,
                k_per_video=config.k_per_video,
                rng=rng,
            )
        elif not new_videos:
            logger.info("Trick '{}' -> 0 new pairs (no new clips)", trick_name)
            continue
        else:
            pairs = _generate_pairs_for_new_videos(
                all_videos=unique_videos,
                new_videos=new_videos,
                existing_videos=existing_videos,
                existing_pairs=existing_pairs,
                k_per_video=config.k_per_video,
                rng=rng,
            )

        start_idx = next_index_by_trick.get(trick_name, 1)
        for offset, (left, right) in enumerate(sorted(pairs), start=0):
            payload = {"data": {"trick": trick_name, "left": left, "right": right}}
            filename = f"{trick_name}__{start_idx + offset:06d}.json"
            out_path = output_dir / filename
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            created_pairs += 1
        next_index_by_trick[trick_name] = start_idx + len(pairs)
        logger.info("Trick '{}' -> {} new pairs", trick_name, len(pairs))

    logger.info(
        "Created {} new pair files in {} ({} tricks skipped)",
        created_pairs,
        output_dir,
        skipped_tricks,
    )

    return created_pairs


def _prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if force:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _generate_pairs_for_all_videos(
    *, unique_videos: list[str], k_per_video: int, rng: random.Random
) -> set[tuple[str, str]]:
    shuffled = list(unique_videos)
    rng.shuffle(shuffled)

    pairs: set[tuple[str, str]] = set()

    for left, right in zip(shuffled, shuffled[1:]):
        pairs.add(_make_pair(left, right))

    for video in shuffled:
        opponents = [other for other in unique_videos if other != video]
        if not opponents:
            continue
        if len(opponents) <= k_per_video:
            picks = opponents
        else:
            picks = rng.sample(opponents, k_per_video)
        for other in picks:
            pairs.add(_make_pair(video, other))

    return pairs


def _generate_pairs_for_new_videos(
    *,
    all_videos: list[str],
    new_videos: list[str],
    existing_videos: set[str],
    existing_pairs: set[tuple[str, str]],
    k_per_video: int,
    rng: random.Random,
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()

    shuffled_new = list(new_videos)
    rng.shuffle(shuffled_new)

    for left, right in zip(shuffled_new, shuffled_new[1:]):
        pair = _make_pair(left, right)
        if pair not in existing_pairs:
            pairs.add(pair)

    for video in shuffled_new:
        remaining_budget = k_per_video

        if existing_videos:
            existing_candidates = sorted(existing_videos - {video})
            if existing_candidates:
                anchor = rng.choice(existing_candidates)
                pair = _make_pair(video, anchor)
                if pair not in existing_pairs:
                    pairs.add(pair)
                    remaining_budget = max(remaining_budget - 1, 0)

        opponents = [other for other in all_videos if other != video]
        unseen_opponents = [
            other
            for other in opponents
            if _make_pair(video, other) not in existing_pairs
            and _make_pair(video, other) not in pairs
        ]

        if not unseen_opponents or remaining_budget <= 0:
            continue

        if len(unseen_opponents) <= remaining_budget:
            picks = unseen_opponents
        else:
            picks = rng.sample(unseen_opponents, remaining_budget)

        for other in picks:
            pairs.add(_make_pair(video, other))

    return pairs


def _load_existing_pairs(
    output_dir: Path,
) -> tuple[dict[str, set[tuple[str, str]]], dict[str, set[str]], dict[str, int]]:
    pairs_by_trick: dict[str, set[tuple[str, str]]] = {}
    videos_by_trick: dict[str, set[str]] = {}
    next_index_by_trick: dict[str, int] = {}
    pattern = re.compile(r"^(?P<trick>.+)__(?P<idx>\d{6})\.json$")

    for path in sorted(output_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        data = payload.get("data") or {}
        trick = data.get("trick")
        left = data.get("left")
        right = data.get("right")

        if not all(isinstance(value, str) and value for value in [trick, left, right]):
            raise RuntimeError(f"Unsupported existing pair file format: {path}")

        trick_name = str(trick)
        pair = _make_pair(left, right)
        pairs_by_trick.setdefault(trick_name, set()).add(pair)
        videos_by_trick.setdefault(trick_name, set()).update([left, right])

        match = pattern.match(path.name)
        if match and match.group("trick") == trick_name:
            idx = int(match.group("idx"))
            next_index_by_trick[trick_name] = max(
                next_index_by_trick.get(trick_name, 1), idx + 1
            )
        else:
            next_index_by_trick[trick_name] = max(
                next_index_by_trick.get(trick_name, 1), 1
            )

    return pairs_by_trick, videos_by_trick, next_index_by_trick


def _make_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)
