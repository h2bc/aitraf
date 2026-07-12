from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from aitraf_core.storage.s3 import (
    copy_s3_object,
    download_s3_uri,
    load_s3_settings,
    parse_s3_uri,
)


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


def test_download_s3_uri_uses_get_object_without_head(tmp_path: Path) -> None:
    class Client:
        def get_object(self, *, Bucket: str, Key: str) -> dict[str, BytesIO]:
            assert (Bucket, Key) == ("aitraf", "clips/sample.mp4")
            return {"Body": BytesIO(b"video")}

    destination = tmp_path / "sample.mp4"
    download_s3_uri(
        "s3://aitraf/clips/sample.mp4", destination, s3_client=Client()
    )

    assert destination.read_bytes() == b"video"


def test_copy_s3_object_uses_server_side_copy() -> None:
    calls: list[dict] = []

    class Client:
        def copy_object(self, **kwargs) -> None:
            calls.append(kwargs)

    copy_s3_object(
        Client(), source_bucket="private", source_key="clips/sample.mp4",
        destination_bucket="public", destination_key="videos/sample.mp4",
    )
    assert calls == [{
        "Bucket": "public", "Key": "videos/sample.mp4",
        "CopySource": {"Bucket": "private", "Key": "clips/sample.mp4"},
    }]


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
