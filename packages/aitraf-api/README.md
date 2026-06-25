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

Endpoint tests mock model prediction and are split by feature under
`packages/aitraf-api/tests`.

## Docker Image

Build the production API image from the repository root:

```bash
docker build -f packages/aitraf-api/Dockerfile -t aitraf-api:local .
```

The image installs `aitraf-api` and `aitraf-core`, includes the small repo
`data/` directory for manifests and vocabularies, and keeps `storage/` external.
It does not include train package code, committed secrets, model artifacts, or
generated storage contents.

Run with explicit runtime settings and mounted storage:

```bash
docker run --rm -p 8000:8000 \
  -e AITRAF_API_TOKEN="$AITRAF_API_TOKEN" \
  -e AITRAF_CLASSIFICATION_MODEL_URI="$AITRAF_CLASSIFICATION_MODEL_URI" \
  -e AITRAF_AQA_MODEL_URI="$AITRAF_AQA_MODEL_URI" \
  -e AITRAF_DATA_PATH=/workspace/data \
  -e AITRAF_STORAGE_PATH=/workspace/storage \
  -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
  -v "$AITRAF_STORAGE_PATH:/workspace/storage:ro" \
  aitraf-api:local
```

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
