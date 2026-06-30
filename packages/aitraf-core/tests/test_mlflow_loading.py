from __future__ import annotations

from types import SimpleNamespace

import torch

import aitraf_core.loading.mlflow as mlflow_loading


class DummyMlflowClient:
    def get_run(self, run_id: str):
        return SimpleNamespace(data=SimpleNamespace(params={"run_id": run_id}))


def test_load_mlflow_torch_model_passes_cpu_map_location(monkeypatch):
    calls: dict[str, str] = {}

    def load_model(model_uri: str, *, map_location: str):
        calls["model_uri"] = model_uri
        calls["map_location"] = map_location
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(mlflow_loading.mlflow.pytorch, "load_model", load_model)
    monkeypatch.setattr(
        mlflow_loading.mlflow.models,
        "get_model_info",
        lambda model_uri: SimpleNamespace(run_id="run-1", metadata={"id2label": {}}),
    )
    monkeypatch.setattr(mlflow_loading, "MlflowClient", DummyMlflowClient)
    mlflow_loading.load_mlflow_torch_model.cache_clear()

    loaded = mlflow_loading.load_mlflow_torch_model("models:/aqa/test", device="cpu")

    assert calls == {"model_uri": "models:/aqa/test", "map_location": "cpu"}
    assert next(loaded.model.parameters()).device.type == "cpu"


def test_mlflow_transformers_device_maps_cpu_to_pipeline_cpu():
    assert mlflow_loading._mlflow_transformers_device("cpu") == -1
