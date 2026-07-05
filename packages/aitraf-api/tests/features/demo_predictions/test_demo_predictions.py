from __future__ import annotations

from fastapi.testclient import TestClient


def test_demo_predictions_returns_prepared_response(
    client: TestClient,
    auth_headers: dict[str, str],
    video_id: str,
) -> None:
    response = client.get("/demo-predictions", headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload == [
        {
            "video_id": video_id,
            "video_url": f"https://s3.example.test/aitraf/clips/{video_id}?signed=true",
            "person": "person-a",
            "ground_truth": {
                "trick": "mizou",
                "execution_score": "3",
            },
            "predictions": {
                "trick_classification": {
                    "label": "mizou",
                    "confidence": 0.91,
                },
                "trick_aqa": {
                    "label": "3",
                    "confidence": 0.82,
                },
            },
        }
    ]
    assert "video_url" in payload[0]
    assert "s3_path" not in payload[0]


def test_demo_predictions_requires_authentication(client: TestClient) -> None:
    response = client.get("/demo-predictions")

    assert response.status_code == 401
    assert response.json() == {"detail": "Missing app token"}
