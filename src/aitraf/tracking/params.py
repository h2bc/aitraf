"""MLflow parameter helpers for evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from mlflow.tracking import MlflowClient


def build_training_params(
    run_id: str,
    param_map: Mapping[str, str],
) -> dict[str, Any]:
    """Return the selected evaluation params from a training run."""

    training_run = MlflowClient().get_run(run_id)
    params = training_run.data.params
    metrics = training_run.data.metrics

    return {
        **{
            output_name: params.get(source_name)
            for output_name, source_name in param_map.items()
        },
        "trained_epochs": metrics.get("epoch"),
    }


def params_to_df(params: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"param": key, "value": value} for key, value in params.items()],
        columns=["param", "value"],
    )


__all__ = [
    "build_training_params",
    "params_to_df",
]
