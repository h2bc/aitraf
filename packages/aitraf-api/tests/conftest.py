from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aitraf_api.features.health import router as health_router


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(health_router)
    return TestClient(app)
