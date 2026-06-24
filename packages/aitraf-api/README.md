# aitraf-api

FastAPI inference surface for the AITRAF demo frontend.

## Environment

Required at app startup:

- `AITRAF_CLASSIFICATION_MODEL_URI`: MLflow model URI for trick classification, for example `models:/aitraf-trick-classification@infant`.
- `AITRAF_AQA_MODEL_URI`: MLflow model URI for temporal-fusion trick AQA, for example `models:/aitraf-trick-aqa-temporal-fusion@infant`.
- `AITRAF_DATA_PATH`: repo data directory; manifests are derived from this.
- `AITRAF_STORAGE_PATH`: repo storage directory; clips and VideoMAE features are derived from this.

Required for protected endpoints:

- `AITRAF_API_TOKEN`: bearer token expected from the frontend.

Required before registered model inference can run:

- `MLFLOW_TRACKING_URI`: MLflow tracking server URI.
- Any MLflow credentials needed by the target deployment.


## Run Locally

```bash
task api:run
```

FastAPI serves interactive Swagger docs at `http://localhost:8000/docs` and
OpenAPI JSON at `http://localhost:8000/openapi.json`.

## Test

```bash
task api:test
```

Tests use the configured MLflow model URIs and are split by feature under
`packages/aitraf-api/tests/features/`. The current retained smoke coverage is
the health endpoint test.

## Temporal-Fusion Trick AQA Smoke

Fetch demo videos, then use one returned `id` for trick AQA:

```bash
curl -H "Authorization: Bearer $AITRAF_API_TOKEN" \
  http://localhost:8000/demo-videos

curl -X POST -H "Authorization: Bearer $AITRAF_API_TOKEN" \
  "http://localhost:8000/inference/trick-aqa/{id}"
```

Expected response shape:

```json
{
  "video_id": "{id}",
  "prediction": {"label": "3", "confidence": 0.72},
  "ground_truth": {"label": "3"},
  "model": {"kind": "video_mae_temporal_fusion"}
}
```

## Architecture Boundary

Runtime video/frame processing is delegated to `aitraf-core`.
