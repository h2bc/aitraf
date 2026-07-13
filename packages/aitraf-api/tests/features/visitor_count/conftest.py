from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aitraf_api.features import router


class FakeVisitorCounter:
    def __init__(self, count: int = 0) -> None:
        self.count = count
        self.validate_calls = 0
        self.close_calls = 0

    async def validate(self) -> None:
        self.validate_calls += 1

    async def increment(self) -> int:
        self.count += 1
        return self.count

    async def aclose(self) -> None:
        self.close_calls += 1


@pytest.fixture()
def visitor_counter() -> FakeVisitorCounter:
    return FakeVisitorCounter(count=41)


@pytest.fixture()
def visitor_client(visitor_counter: FakeVisitorCounter) -> AsyncIterator[TestClient]:
    app = FastAPI()
    app.state.settings = SimpleNamespace(api_token="protected-token")
    app.state.visitor_counter = visitor_counter
    app.include_router(router)
    with TestClient(app) as client:
        yield client
