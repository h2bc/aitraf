"""Runtime settings for the AITRAF API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    api_token: str
    classification_predictions_run_id: str
    aqa_predictions_run_id: str
    aws_endpoint_url: str
    aws_bucket: str


def load_settings(
    env: Mapping[str, str],
    *,
    root: Path,
) -> Settings:
    load_dotenv(root / ".env", override=False)

    return Settings(
        api_token=env["AITRAF_API_TOKEN"],
        classification_predictions_run_id=env[
            "AITRAF_CLASSIFICATION_PREDICTIONS_RUN_ID"
        ],
        aqa_predictions_run_id=env["AITRAF_AQA_PREDICTIONS_RUN_ID"],
        aws_endpoint_url=env["AWS_ENDPOINT_URL"],
        aws_bucket=env["AWS_BUCKET"],
    )


__all__ = [
    "Settings",
    "load_settings",
]
