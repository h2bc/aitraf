"""MLflow model loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from torch import nn


@dataclass(frozen=True)
class LoadedTransformersModel:
    model: nn.Module
    image_processor: Any
    run_id: str
    run_params: Mapping[str, str]


def load_transformers_model(
    model_uri: str,
    *,
    tracking_uri: str | None = None,
) -> LoadedTransformersModel:
    import mlflow
    from mlflow import transformers as mlflow_transformers
    from mlflow.tracking import MlflowClient

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    components = mlflow_transformers.load_model(
        model_uri,
        return_type="components",
    )
    model_info = mlflow.models.get_model_info(model_uri)
    run = MlflowClient().get_run(model_info.run_id)

    return LoadedTransformersModel(
        model=components["model"],
        image_processor=components["image_processor"],
        run_id=model_info.run_id,
        run_params=run.data.params,
    )


__all__ = [
    "LoadedTransformersModel",
    "load_transformers_model",
]
