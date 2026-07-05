"""FastAPI response schemas for the demo predictions API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]


class GroundTruth(BaseModel):
    trick: str
    execution_score: str


class TaskPrediction(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class TaskPredictions(BaseModel):
    trick_classification: TaskPrediction
    trick_aqa: TaskPrediction


class DemoPrediction(BaseModel):
    video_id: str
    video_url: str
    person: str
    ground_truth: GroundTruth
    predictions: TaskPredictions


class ErrorResponse(BaseModel):
    detail: str


__all__ = [
    "DemoPrediction",
    "ErrorResponse",
    "GroundTruth",
    "HealthResponse",
    "TaskPrediction",
    "TaskPredictions",
]
