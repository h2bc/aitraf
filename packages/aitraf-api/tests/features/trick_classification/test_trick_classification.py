from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aitraf_api.features.trick_classification import service as service_module


def test_trick_classification_returns_prediction(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    video_id: str,
) -> None:
    monkeypatch.setattr(
        service_module,
        "predict_trick_classification_video_mae",
        lambda **kwargs: ("mizou", 0.91),
    )

    response = client.post(
        f"/inference/trick-classification/{video_id}",
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "video_id": video_id,
        "prediction": {"label": "mizou", "confidence": 0.91},
        "ground_truth": {"label": "mizou"},
        "model": {"kind": "video_mae"},
    }
