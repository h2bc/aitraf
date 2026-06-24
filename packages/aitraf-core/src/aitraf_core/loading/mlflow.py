"""MLflow model loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import mlflow
import mlflow.pytorch
from mlflow import transformers as mlflow_transformers
from mlflow.tracking import MlflowClient
from torch import nn


@dataclass(frozen=True)
class MlflowTransformersModel:
    model: nn.Module
    image_processor: Any
    run_id: str
    run_params: Mapping[str, str]


@dataclass(frozen=True)
class MlflowTorchModel:
    model: nn.Module
    run_id: str
    run_params: Mapping[str, str]


@lru_cache(maxsize=4)
def load_mlflow_transformers_model(
    model_uri: str,
) -> MlflowTransformersModel:
    components = mlflow_transformers.load_model(
        model_uri,
        return_type="components",
    )
    model_info = mlflow.models.get_model_info(model_uri)
    run = MlflowClient().get_run(model_info.run_id)

    return MlflowTransformersModel(
        model=components["model"],
        image_processor=components["image_processor"],
        run_id=model_info.run_id,
        run_params=run.data.params,
    )


@lru_cache(maxsize=4)
def load_mlflow_torch_model(
    model_uri: str,
) -> MlflowTorchModel:
    model = mlflow.pytorch.load_model(model_uri)
    model_info = mlflow.models.get_model_info(model_uri)
    run = MlflowClient().get_run(model_info.run_id)

    return MlflowTorchModel(
        model=model,
        run_id=model_info.run_id,
        run_params=run.data.params,
    )


__all__ = [
    "MlflowTorchModel",
    "MlflowTransformersModel",
    "load_mlflow_torch_model",
    "load_mlflow_transformers_model",
]
