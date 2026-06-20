from __future__ import annotations

from fastapi.testclient import TestClient


def test_classification_returns_prediction_for_valid_video_id(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.post(
        "/inference/trick-classification/shared.mp4",
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "video_id": "shared.mp4",
        "prediction": {"label": "top-soul", "confidence": 0.91},
        "ground_truth": {"label": "top-soul"},
        "model": {"kind": "video_mae"},
    }
