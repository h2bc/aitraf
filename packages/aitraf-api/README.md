# aitraf-api

FastAPI inference surface for the AITRAF demo frontend.

## Environment

Required at app startup:

- `AITRAF_CLASSIFICATION_MODEL_URI`: MLflow model URI for trick classification, for example `models:/aitraf-trick-classification@infant`.
- `AITRAF_AQA_MODEL_URI`: MLflow model URI for trick AQA, for example `models:/aitraf-trick-aqa@infant`.
- `AITRAF_DATA_PATH`: repo data directory; manifests are derived from this.
- `AITRAF_STORAGE_PATH`: repo storage directory; clips are derived from this.

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

Tests use stubbed model loading/prediction and are split by feature under
`packages/aitraf-api/tests/features/`. Shared auth coverage lives in
`packages/aitraf-api/tests/test_auth.py`.

## Architecture Boundary

Runtime video/frame processing is delegated to `aitraf-core`.
