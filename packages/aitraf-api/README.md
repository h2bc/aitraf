# aitraf-api

FastAPI inference surface for the AITRAF demo frontend.

## Environment

Required at app startup:

- `AITRAF_CLASSIFICATION_MODEL_URI`: MLflow model URI for trick classification, for example `models:/aitraf-trick-classification@infant`.
- `AITRAF_AQA_MODEL_URI`: MLflow model URI for temporal-fusion trick AQA, for example `models:/aitraf-trick-aqa-temporal-fusion@infant`.
- `AITRAF_API_DEVICE`: inference device policy: `auto`, `cpu`, or `cuda`.
- `AITRAF_DATA_PATH`: repo data directory; manifests are derived from this.
- `AITRAF_STORAGE_PATH`: repo storage directory; clips and VideoMAE features are derived from this.

Required for protected endpoints:

- `AITRAF_API_TOKEN`: bearer token expected from the frontend.

Required before registered model inference can run:

- `MLFLOW_TRACKING_URI`: MLflow tracking server URI.
- Any MLflow credentials needed by the target deployment.

Optional for demo clip hydration at startup:

- `AITRAF_API_DEMO_CLIPS_DOWNLOAD`: set to `1`, `true`, `yes`, or `on` to download the currently selected demo clips into `AITRAF_STORAGE_PATH/data/clips` before the API finishes startup.
- `AITRAF_API_DEMO_CLIPS_FORCE_DOWNLOAD`: set to `1`, `true`, `yes`, or `on` to re-download demo clips even when they already exist locally.
- `AWS_ENDPOINT_URL`, `AWS_DEFAULT_REGION`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY`: required when demo clip download is enabled.


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

The image installs `aitraf-api` and `aitraf-core` and includes the small repo
`data/` directory for manifests and vocabularies. It does not include train
package code, committed secrets, model artifacts, feature caches, clips, or full
storage contents; the root `.dockerignore` keeps `storage/` out of the build
context. Demo clips can be hydrated at runtime with `AITRAF_API_DEMO_CLIPS_DOWNLOAD=1`.

Run with explicit runtime settings and mounted storage:

```bash
docker run --rm -p 8000:8000 \
  -e AITRAF_API_TOKEN="$AITRAF_API_TOKEN" \
  -e AITRAF_CLASSIFICATION_MODEL_URI="$AITRAF_CLASSIFICATION_MODEL_URI" \
  -e AITRAF_AQA_MODEL_URI="$AITRAF_AQA_MODEL_URI" \
  -e AITRAF_API_DEVICE="$AITRAF_API_DEVICE" \
  -e AITRAF_DATA_PATH=/workspace/data \
  -e AITRAF_STORAGE_PATH=/workspace/storage \
  -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
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
