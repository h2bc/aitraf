from __future__ import annotations

from pathlib import Path

import pytest

from aitraf_api.config import load_settings


def _env() -> dict[str, str]:
    return {
        "AITRAF_API_TOKEN": "token",
        "AITRAF_CLASSIFICATION_PREDICTIONS_RUN_ID": "classification",
        "AITRAF_AQA_PREDICTIONS_RUN_ID": "aqa",
        "AWS_ENDPOINT_URL": "https://s3.test",
        "AWS_BUCKET": "private",
        "AITRAF_PUBLIC_ASSET_BUCKET": "public",
        "AITRAF_REDIS_URL": "redis://localhost:6379/0",
    }


def test_load_settings_requires_redis_url(tmp_path: Path) -> None:
    env = _env()
    del env["AITRAF_REDIS_URL"]

    with pytest.raises(KeyError, match="AITRAF_REDIS_URL"):
        load_settings(env, root=tmp_path)


def test_load_settings_reads_redis_url(tmp_path: Path) -> None:
    settings = load_settings(_env(), root=tmp_path)

    assert settings.redis_url == "redis://localhost:6379/0"
