from __future__ import annotations

import pytest

from aitraf_api.config import load_settings, resolve_api_device


def test_resolve_api_device_auto_uses_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr("aitraf_api.config.torch.cuda.is_available", lambda: False)

    assert resolve_api_device("auto") == "cpu"


def test_resolve_api_device_auto_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr("aitraf_api.config.torch.cuda.is_available", lambda: True)

    assert resolve_api_device("auto") == "cuda"


def test_resolve_api_device_cuda_fails_without_cuda(monkeypatch):
    monkeypatch.setattr("aitraf_api.config.torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="AITRAF_API_DEVICE=cuda"):
        resolve_api_device("cuda")


def test_resolve_api_device_rejects_unknown_value():
    with pytest.raises(RuntimeError, match="AITRAF_API_DEVICE must be one of"):
        resolve_api_device("gpu")


def test_load_settings_reads_api_demo_clip_flags(tmp_path, monkeypatch):
    monkeypatch.setattr("aitraf_api.config.torch.cuda.is_available", lambda: False)
    env = {
        "AITRAF_API_DEVICE": "auto",
        "AITRAF_DATA_PATH": str(tmp_path / "data"),
        "AITRAF_STORAGE_PATH": str(tmp_path / "storage"),
        "AITRAF_API_TOKEN": "token",
        "AITRAF_CLASSIFICATION_MODEL_URI": "models:/classification@demo",
        "AITRAF_AQA_MODEL_URI": "models:/aqa@demo",
        "AITRAF_API_DEMO_CLIPS_DOWNLOAD": "1",
        "AITRAF_API_DEMO_CLIPS_FORCE_DOWNLOAD": "true",
    }

    settings = load_settings(env, root=tmp_path)

    assert settings.demo_clips.download_enabled is True
    assert settings.demo_clips.force_download is True
