"""Response schemas for health checks."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]


__all__ = ["HealthResponse"]
