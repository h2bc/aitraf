from __future__ import annotations

from fastapi.testclient import TestClient


def test_aqa_returns_prediction_for_valid_video_id(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.post(
        "/inference/trick-aqa/shared.mp4",
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "video_id": "shared.mp4",
        "prediction": {"label": "3", "confidence": 0.7506087422370911},
        "ground_truth": {"label": "3"},
        "model": {"kind": "video_mae_temporal_fusion"},
    }
