from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aitraf_api.features.trick_assessment import service as service_module


def test_trick_assessment_returns_prediction(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    video_id: str,
) -> None:
    monkeypatch.setattr(
        service_module,
        "predict_trick_assessment_video_mae_temporal_fusion",
        lambda **kwargs: ("3", 0.82),
    )

    response = client.post(
        f"/inference/trick-aqa/{video_id}",
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "video_id": video_id,
        "prediction": {"label": "3", "confidence": 0.82},
        "ground_truth": {"label": "3"},
        "model": {"kind": "video_mae_temporal_fusion"},
    }
