# Quickstart: API Inference Surface

## Prerequisites

- Workspace dependencies installed with `uv sync`.
- Current test manifests present:
  - `data/manifests/trick_classification/test.jsonl`
  - `data/manifests/score_prediction_ordinal/test.jsonl`
- `AITRAF_DATA_PATH` points at the repo data directory, normally `data`.
- `AITRAF_STORAGE_PATH` points at the repo storage directory, normally `storage`.
- Local clips available at `$AITRAF_STORAGE_PATH/data/clips`.
- API token provided through environment/configuration.
- MLflow model URIs configured through environment variables:
  - `AITRAF_CLASSIFICATION_MODEL_URI`
  - `AITRAF_AQA_MODEL_URI`
- Runtime config provides endpoint-to-model URI mapping and derives manifest/clip paths from the configured data and storage roots.
- MLflow provides the trained model artifact and logged image processor.

## Validation Scope

The API tests should prove basic service behavior only:

- Health endpoint responds.
- Protected endpoints reject missing or invalid tokens.
- Demo-video endpoint returns a flat display list of videos and metadata from the current manifests.
- Inference endpoints accept only an id and call the inference service for valid ids.

Do not validate model performance, model quality, metric values, latency targets,
or exhaustive edge cases in this feature's API test suite.

## Run Tests

```bash
task api:test
```

Expected outcome: focused API tests pass using pytest fixtures and
Arrange/Act/Assert structure. Stub predictors should keep tests from downloading
or loading large model weights.

## Run API Locally

```bash
export AITRAF_API_TOKEN="replace-with-local-token"
export MLFLOW_TRACKING_URI="replace-with-mlflow-uri"
export AITRAF_CLASSIFICATION_MODEL_URI="models:/aitraf-trick-classification@infant"
export AITRAF_AQA_MODEL_URI="models:/aitraf-trick-aqa@infant"
export AITRAF_DATA_PATH="data"
export AITRAF_STORAGE_PATH="storage"
task api:run
```

Expected outcome: FastAPI starts and exposes Swagger docs at
`http://localhost:8000/docs`.

## Smoke Requests

Health has no parameters:

```bash
curl http://localhost:8000/health
```

Demo videos has no parameters and requires the token:

```bash
curl -H "Authorization: Bearer $AITRAF_API_TOKEN" \
  http://localhost:8000/demo-videos
```

Inference takes only the id returned in the flat `videos` list from `GET /demo-videos`:

```bash
curl -X POST -H "Authorization: Bearer $AITRAF_API_TOKEN" \
  http://localhost:8000/inference/trick-classification/{id}

curl -X POST -H "Authorization: Bearer $AITRAF_API_TOKEN" \
  http://localhost:8000/inference/trick-aqa/{id}
```

Expected outcome: valid ids run the configured video-to-frames, image processor,
and model prediction path, then return the selected video id, user-formatted
prediction with confidence, user-formatted ground truth, and compact model
metadata. Missing videos, missing runtime config, missing model artifacts, or
invalid ids return explicit errors rather than dummy predictions.

## Related Artifacts

- API contract: `specs/002-api-inference/contracts/openapi.yaml`
- Data model: `specs/002-api-inference/data-model.md`
- Implementation plan: `specs/002-api-inference/plan.md`
