from __future__ import annotations

from fastapi.testclient import TestClient


def test_demo_videos_returns_matching_rows(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.get("/demo-videos", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == {
        "videos": [
            {
                "id": "shared.mp4",
                "video_id": "shared.mp4",
                "s3_path": "s3://aitraf/clips/shared.mp4",
                "person": "Justas",
                "trick": "top-soul",
                "execution_score": "3",
            }
        ]
    }
