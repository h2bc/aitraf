#!/usr/bin/env python3
"""
Download a Label Studio project export and store it as parquet.

Usage:
    python -m scripts.pull_ls [--project-id 123] [--output data/label_studio_export.parquet]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from label_studio_sdk import LabelStudio

from paths import DATA_DIR

EXPECTED_COLUMNS = ["video", "trick", "key_foot", "person"]


def get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a Label Studio export and save it as parquet."
    )
    parser.add_argument(
        "--project-id",
        type=int,
        default=None,
        help="Label Studio project ID (defaults to LABEL_STUDIO_PROJECT_ID env var).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "label_studio_export.parquet",
        help="Destination parquet file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    base_url = get_env("LABEL_STUDIO_URL")
    token = get_env("LABEL_STUDIO_TOKEN").strip()
    project_id = args.project_id or int(get_env("LABEL_STUDIO_PROJECT_ID"))

    if args.output.exists() and not args.force:
        raise SystemExit(
            f"Output file {args.output} already exists. Use --force to overwrite."
        )

    client = LabelStudio(base_url=base_url, api_key=token)

    print(f"Fetching export for project {project_id} …")
    df = client.projects.exports.as_pandas(project_id)

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise SystemExit(
            f"Export missing expected columns: {', '.join(missing_cols)}. "
            "Did the project schema change?"
        )

    df = df.dropna(subset=["video"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(
        f"Saved {len(df):,} labeled rows with columns "
        f"{', '.join(EXPECTED_COLUMNS)} to {args.output}"
    )


if __name__ == "__main__":
    main()
