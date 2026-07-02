# aitraf-api

FastAPI demo API that serves precomputed AITRAF predictions from MLflow
artifacts. The API does not load models or run live inference.

## Environment

Required at app startup:

- `AITRAF_API_TOKEN`: bearer token for protected endpoints.
- `AITRAF_CLASSIFICATION_PREDICTIONS_RUN_ID`: MLflow run containing classification `test_predictions.json`.
- `AITRAF_AQA_PREDICTIONS_RUN_ID`: MLflow run containing AQA `test_predictions.json`.
- `MLFLOW_TRACKING_URI`: MLflow tracking server URI.
- Any MLflow credentials required by the deployment.
- S3-compatible artifact store credentials required by MLflow artifact download,
  such as `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`, and
  `AWS_DEFAULT_REGION`. For S3-compatible artifact stores, pass
  `MLFLOW_S3_ENDPOINT_URL="${AWS_ENDPOINT_URL}"` as well.

Validation run IDs:

```bash
export AITRAF_CLASSIFICATION_PREDICTIONS_RUN_ID="2b2208e417e34e2198bb108e4f683cf9"
export AITRAF_AQA_PREDICTIONS_RUN_ID="da6a8082c5e646448c7a79cd124b8e09"
```

The artifact path is fixed in code as `test_predictions.json`; do not configure
artifact paths through environment variables.

## Run Locally

```bash
task api:run
```

On startup the API downloads both prediction artifacts, matches rows by
`video_id`, and stores the final response in memory.

## Endpoint

```bash
curl --fail \
  -H "Authorization: Bearer ${AITRAF_API_TOKEN}" \
  http://localhost:8000/demo-predictions
```

The response is a JSON array. Each item contains demo video metadata plus:

- `predictions.trick_classification`
- `predictions.trick_aqa`

## Test

```bash
uv run --package aitraf-api pytest packages/aitraf-api/tests/features/demo_predictions
```

The endpoint tests cover the successful response shape and missing
authentication.

## Docker Image

Build from the repository root:

```bash
docker build -f packages/aitraf-api/Dockerfile -t aitraf-api:precomputed .
```

Run with MLflow credentials and prediction run IDs:

```bash
docker run --rm -p 8000:8000 \
  -e AITRAF_API_TOKEN="${AITRAF_API_TOKEN}" \
  -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
  -e MLFLOW_TRACKING_USERNAME="${MLFLOW_TRACKING_USERNAME}" \
  -e MLFLOW_TRACKING_PASSWORD="${MLFLOW_TRACKING_PASSWORD}" \
  -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
  -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
  -e AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL}" \
  -e MLFLOW_S3_ENDPOINT_URL="${AWS_ENDPOINT_URL}" \
  -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \
  -e AITRAF_CLASSIFICATION_PREDICTIONS_RUN_ID="2b2208e417e34e2198bb108e4f683cf9" \
  -e AITRAF_AQA_PREDICTIONS_RUN_ID="da6a8082c5e646448c7a79cd124b8e09" \
  aitraf-api:precomputed
```

The image is CPU-only and does not install `aitraf-core`, Torch, Transformers,
CUDA dependencies, or ffmpeg.
