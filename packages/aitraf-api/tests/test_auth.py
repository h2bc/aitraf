from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize(
    ("headers", "expected_detail"),
    [
        ({}, "Missing app token"),
        ({"Authorization": "Bearer wrong-token"}, "Invalid app token"),
    ],
)
def test_demo_videos_rejects_missing_or_invalid_token(
    client: TestClient,
    headers: dict[str, str],
    expected_detail: str,
) -> None:
    response = client.get("/demo-videos", headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": expected_detail}
