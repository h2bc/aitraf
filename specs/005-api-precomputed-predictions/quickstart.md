# Quickstart: AITRAF API Precomputed Predictions

This validates the simplified API after implementation.

## Runtime Config

The API needs an auth token and two prediction run IDs that contain full
`test_predictions.json` artifacts. Prediction artifact paths are fixed constants
in API code, not `.env` values:

```bash
export AITRAF_API_TOKEN="demo-token"
export AITRAF_CLASSIFICATION_PREDICTIONS_RUN_ID="2b2208e417e34e2198bb108e4f683cf9"
export AITRAF_AQA_PREDICTIONS_RUN_ID="da6a8082c5e646448c7a79cd124b8e09"
```

## Validation Prediction Runs

These supplied MLflow eval runs contain downloadable full `test_predictions.json`
artifacts and are used for real API/Docker smoke validation:

```bash
export AITRAF_CLASSIFICATION_PREDICTIONS_RUN_ID="2b2208e417e34e2198bb108e4f683cf9"
export AITRAF_AQA_PREDICTIONS_RUN_ID="da6a8082c5e646448c7a79cd124b8e09"
```

Verified artifact contents:

- Classification run `2b2208e417e34e2198bb108e4f683cf9`: `test_predictions.json`, 100 rows, includes `video_id`, `s3_path`, `person`, `trick`, `execution_score`, and `label`.
- AQA run `da6a8082c5e646448c7a79cd124b8e09`: `test_predictions.json`, 100 rows, includes `video_id`, `s3_path`, `person`, `trick`, `execution_score`, and `label`.

Planning-time inspected run IDs that do not contain full prediction artifacts:

```bash
export AITRAF_CLASSIFICATION_PREDICTIONS_RUN_ID="0a232293f2d44976abc5830953b06a76"
export AITRAF_AQA_PREDICTIONS_RUN_ID="e50e861919cc4187bc4f58c5ab9dc717"
```

Do not use those inspected run IDs for the simplified API. Provide run IDs from
external evaluation runs that contain full `test_predictions.json` artifacts
before running real MLflow/Docker smoke validation.

Reference registered model URIs for external evaluation:

```bash
AITRAF_CLASSIFICATION_MODEL_URI="models:/aitraf-trick-classification@infant"
AITRAF_AQA_MODEL_URI="models:/aitraf-trick-aqa@infant"
```

The API should not require live model config:

```bash
unset AITRAF_CLASSIFICATION_MODEL_URI
unset AITRAF_AQA_MODEL_URI
unset AITRAF_CLASSIFICATION_PREDICTIONS_ARTIFACT_PATH
unset AITRAF_AQA_PREDICTIONS_ARTIFACT_PATH
```

## Tests

```bash
uv run --package aitraf-api pytest packages/aitraf-api/tests/features/demo_predictions
```

Expected coverage:

- authenticated `/demo-predictions` returns demo prediction records with
  matching classification and AQA prediction rows
- unauthenticated `/demo-predictions` fails with `401`

No `aitraf-train` tests are required for this API simplification. Train-side
prediction export changes are validated by inspecting the supplied MLflow
artifacts.

## Smoke Run

```bash
uv run --package aitraf-api uvicorn aitraf_api.app:create_app --factory --host 0.0.0.0 --port 8000
```

```bash
curl --fail http://localhost:8000/health
```

```bash
curl --fail \
  -H "Authorization: Bearer ${AITRAF_API_TOKEN}" \
  http://localhost:8000/demo-predictions
```

Expected result:

- response is a JSON array
- each record has `predictions.trick_classification`
- each record has `predictions.trick_aqa`
- no model inference happens

## Docker Check

```bash
docker build -f packages/aitraf-api/Dockerfile -t aitraf-api:precomputed .
```

Expected result:

- no CUDA base image
- no ffmpeg install
- no `aitraf-core` copy/install
- no `torch` or `transformers` dependency in the API package

## Real Docker Smoke Test

After implementation, run the built image with supplied run IDs that contain
full prediction artifacts. This is the end-to-end validation path: the API
container must download both `test_predictions.json` artifacts from MLflow at
startup, match them by `video_id`, cache the final response in memory, and serve
that response from `GET /demo-predictions`.

Do not copy prediction artifacts into the image or mount local prediction files
for this smoke test. The only prediction inputs are the two MLflow run IDs:

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

Then verify:

```bash
curl --fail http://localhost:8000/health
curl --fail \
  -H "Authorization: Bearer ${AITRAF_API_TOKEN}" \
  http://localhost:8000/demo-predictions
```

Expected result:

- the image starts without live model config
- the API downloads `test_predictions.json` from both configured MLflow runs
- startup fails loudly if either artifact cannot be downloaded or parsed
- `/demo-predictions` returns cached precomputed demo prediction records from
  the in-memory startup result
- no live inference dependencies or routes are required

Current validation status:

- Local API startup with the supplied MLflow run IDs succeeded and built a
  matched demo response with 10 records.
- API demo-predictions endpoint tests passed: `2 passed`.
- Docker image build passed for `aitraf-api:precomputed`.
- Real Docker startup downloaded both MLflow `test_predictions.json` artifacts
  and served `/health` plus authenticated `/demo-predictions`.
- In this workspace, the published host port returned connection refused even
  though Docker reported the mapping, so endpoint smoke was verified through the
  container's Docker bridge IP.
