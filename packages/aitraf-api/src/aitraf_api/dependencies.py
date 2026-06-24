"""Shared FastAPI dependency helpers."""

from __future__ import annotations

from typing import Any

from fastapi import Request

from aitraf_api.config import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_classification_model(request: Request) -> Any:
    return request.app.state.classification_model


def get_aqa_model(request: Request) -> Any:
    return request.app.state.aqa_model


def get_aqa_feature_extractor(request: Request) -> Any:
    return request.app.state.aqa_feature_extractor


def get_aqa_pre_processing(request: Request) -> Any:
    return request.app.state.aqa_pre_processing


__all__ = [
    "get_aqa_feature_extractor",
    "get_aqa_model",
    "get_aqa_pre_processing",
    "get_classification_model",
    "get_settings",
]
