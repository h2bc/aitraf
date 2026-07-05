"""Health service."""

from __future__ import annotations

from aitraf_api.features.health.schemas import HealthResponse


def get_health() -> HealthResponse:
    return HealthResponse(status="ok")


__all__ = ["get_health"]
