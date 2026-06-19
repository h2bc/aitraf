"""FastAPI response schemas for the demo inference API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]


class DemoVideo(BaseModel):
    id: str
    video_id: str
    s3_path: str | None = None
    person: str | None = None
    trick: str | None = None
    execution_score: str | int | float | None = None


class DemoVideosResponse(BaseModel):
    videos: list[DemoVideo]


class PredictionResult(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class DisplayResult(BaseModel):
    label: str


class ModelInfo(BaseModel):
    kind: str


class InferenceResult(BaseModel):
    video_id: str
    prediction: PredictionResult
    ground_truth: DisplayResult
    model: ModelInfo


class ErrorResponse(BaseModel):
    detail: str


__all__ = [
    "DemoVideo",
    "DemoVideosResponse",
    "DisplayResult",
    "ErrorResponse",
    "HealthResponse",
    "InferenceResult",
    "ModelInfo",
    "PredictionResult",
]
