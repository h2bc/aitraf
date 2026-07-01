from __future__ import annotations

import pytest

from aitraf_core.storage.s3 import load_s3_settings, parse_s3_uri


def test_parse_s3_uri_returns_bucket_and_key() -> None:
    assert parse_s3_uri("s3://aitraf/clips/sample.mp4") == (
        "aitraf",
        "clips/sample.mp4",
    )


@pytest.mark.parametrize(
    "uri",
    [
        "https://aitraf/clips/sample.mp4",
        "s3:///clips/sample.mp4",
        "s3://aitraf",
    ],
)
def test_parse_s3_uri_rejects_unsupported_uri(uri: str) -> None:
    with pytest.raises(ValueError):
        parse_s3_uri(uri)


def test_load_s3_settings_requires_expected_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    with pytest.raises(RuntimeError, match="AWS_ENDPOINT_URL"):
        load_s3_settings(require_bucket=False)


def test_load_s3_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://s3.example.test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_BUCKET", "aitraf")

    settings = load_s3_settings(require_bucket=True)

    assert settings.endpoint_url == "https://s3.example.test"
    assert settings.region_name == "us-east-1"
    assert settings.access_key == "access"
    assert settings.secret_key == "secret"
    assert settings.bucket == "aitraf"


def test_load_s3_settings_strips_docker_env_file_quotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://s3.example.test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", '"secret"')

    settings = load_s3_settings(require_bucket=False)

    assert settings.secret_key == "secret"
