from __future__ import annotations

import pytest

from aitraf_api.config import resolve_api_device


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
