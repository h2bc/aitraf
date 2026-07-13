"""Visitor-count response schemas."""

from pydantic import BaseModel, ConfigDict, NonNegativeInt


class VisitorCountResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: NonNegativeInt


__all__ = ["VisitorCountResponse"]
