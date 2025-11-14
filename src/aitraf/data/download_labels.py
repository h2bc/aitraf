"""Label Studio ingestion utility."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from label_studio_sdk import LabelStudio

from aitraf.data import schema
from aitraf.logging import logger


@dataclass
class LabelStudioExportConfig:
    """User-friendly configuration for downloading a Label Studio export."""

    output_path: Path | str
    force: bool = False

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)


def download_labels(config: LabelStudioExportConfig) -> Path:
    """Download a Label Studio export based on env/config settings."""
    load_dotenv()

    output_path = config.output_path
    if output_path.exists() and not config.force:
        logger.info("Labels already exist at {}; skipping", output_path)
        return output_path

    base_url = os.getenv("LABEL_STUDIO_URL")
    token = os.getenv("LABEL_STUDIO_TOKEN")
    project_id = os.getenv("LABEL_STUDIO_PROJECT_ID")

    if not base_url or not token or not project_id:
        raise RuntimeError(
            "LABEL_STUDIO_URL/TOKEN/PROJECT_ID must be set in the environment."
        )

    client = LabelStudio(base_url=base_url, api_key=token.strip())
    df: pd.DataFrame = client.projects.exports.as_pandas(int(project_id))

    missing_cols = [col for col in schema.EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise RuntimeError(
            f"Export missing expected columns: {', '.join(missing_cols)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(
        output_path,
        orient="records",
        lines=True,
        force_ascii=False,
    )
    logger.info("Saved {} labeled rows to {}", len(df), output_path)
    return output_path
