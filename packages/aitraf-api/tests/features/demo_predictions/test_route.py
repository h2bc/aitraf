from fastapi.testclient import TestClient


def test_demo_predictions_returns_stable_public_urls(
    client: TestClient, auth_headers: dict[str, str], video_id: str
) -> None:
    first = client.get("/demo-predictions", headers=auth_headers)
    second = client.get("/demo-predictions", headers=auth_headers)
    assert first.status_code == 200
    payload = first.json()[0]
    assert payload["video_url"] == f"https://s3.example.test/aitraf-public/videos/{video_id}"
    assert payload["thumbnail_url"] == "https://s3.example.test/aitraf-public/thumbnails/sample.jpg"
    assert first.json() == second.json()
    assert "?" not in payload["video_url"]
    assert "s3_path" not in payload
    assert "thumbnail_s3_path" not in payload


def test_demo_predictions_requires_authentication(client: TestClient) -> None:
    response = client.get("/demo-predictions")
    assert response.status_code == 401
