from __future__ import annotations

from fastapi.testclient import TestClient


def test_demo_predictions_returns_predictions_with_presigned_video_url(
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
            "video_url": f"https://s3.example.test/aitraf/clips/{video_id}?signed=1",
            "thumbnail_url": (
                "https://s3.example.test/aitraf/thumbnails/sample.jpg?signed=2"
            ),
            "person": "person-a",
            "key_foot": "right",
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
    assert "thumbnail_s3_path" not in payload[0]

    next_response = client.get("/demo-predictions", headers=auth_headers)
    assert next_response.json()[0]["video_url"] == (
        f"https://s3.example.test/aitraf/clips/{video_id}?signed=3"
    )


def test_demo_predictions_presigns_thumbnail_when_artifact_provides_it(
    client: TestClient,
    auth_headers: dict[str, str],
    classification_prediction_rows: list[dict],
    video_id: str,
) -> None:
    classification_prediction_rows[0]["thumbnail_s3_path"] = (
        f"s3://aitraf/thumbnails/custom-{video_id}.jpg"
    )

    response = client.get("/demo-predictions", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()[0]["thumbnail_url"] == (
        f"https://s3.example.test/aitraf/thumbnails/custom-{video_id}.jpg?signed=2"
    )


def test_demo_predictions_requires_authentication(client: TestClient) -> None:
    response = client.get("/demo-predictions")

    assert response.status_code == 401
    assert response.json() == {"detail": "Missing app token"}
