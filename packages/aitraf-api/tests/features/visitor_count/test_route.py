from fastapi.testclient import TestClient

from aitraf_api.features.visitor_count.service import (
    InvalidVisitorCountError,
    VisitorCountUnavailableError,
)


def test_post_increments_and_returns_result(
    visitor_client: TestClient, visitor_counter: object
) -> None:
    first = visitor_client.post("/visitor-count")
    second = visitor_client.post("/visitor-count")

    assert first.status_code == 200
    assert first.json() == {"count": 42}
    assert second.json() == {"count": 43}
    assert getattr(visitor_counter, "count") == 43


def test_endpoint_is_public(visitor_client: TestClient) -> None:
    response = visitor_client.post("/visitor-count")

    assert response.status_code == 200


def test_body_is_rejected_without_increment(
    visitor_client: TestClient, visitor_counter: object
) -> None:
    response = visitor_client.post("/visitor-count", json={"page": "home"})

    assert response.status_code == 422
    assert getattr(visitor_counter, "count") == 41


def test_query_is_rejected_without_increment(
    visitor_client: TestClient, visitor_counter: object
) -> None:
    response = visitor_client.post("/visitor-count?page=home")

    assert response.status_code == 422
    assert getattr(visitor_counter, "count") == 41


def test_existing_protected_route_remains_protected(visitor_client: TestClient) -> None:
    response = visitor_client.get("/demo-predictions")

    assert response.status_code == 401


class FailingCounter:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def increment(self) -> int:
        raise self.error


def test_unavailable_counter_returns_503(visitor_client: TestClient) -> None:
    visitor_client.app.state.visitor_counter = FailingCounter(
        VisitorCountUnavailableError("Redis visitor count is unavailable")
    )

    response = visitor_client.post("/visitor-count")

    assert response.status_code == 503


def test_invalid_counter_returns_500(visitor_client: TestClient) -> None:
    visitor_client.app.state.visitor_counter = FailingCounter(
        InvalidVisitorCountError("Redis returned an invalid visitor count")
    )

    response = visitor_client.post("/visitor-count")

    assert response.status_code == 500
