"""Response schemas for demo predictions."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
    thumbnail_url: str
    person: str
    key_foot: str
    ground_truth: GroundTruth
    predictions: TaskPredictions


__all__ = [
    "DemoPrediction",
    "GroundTruth",
    "TaskPrediction",
    "TaskPredictions",
]
