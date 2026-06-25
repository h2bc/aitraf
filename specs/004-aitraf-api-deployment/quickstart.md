# Quickstart: AITRAF API Deployment

## Prerequisites

- Docker with Buildx support
- Repository dependencies available through `uv`
- Access to GHCR for workflow publishing
- Runtime API environment values:
  - `AITRAF_API_TOKEN`
  - `AITRAF_CLASSIFICATION_MODEL_URI`
  - `AITRAF_AQA_MODEL_URI`
  - `AITRAF_DATA_PATH`
  - `AITRAF_STORAGE_PATH`
  - `MLFLOW_TRACKING_URI`
  - Any MLflow/artifact credentials required by the deployment

## Validate API Tests

Run the package tests that gate API image publishing:

```bash
task api:test
```

Expected outcome: API tests pass using the API package environment. The GitHub
workflow runs the equivalent package-scoped command:

```bash
uv run --package aitraf-api --extra dev pytest packages/aitraf-api/tests
```

If API tests fail in GitHub Actions, the API image must not publish.

## Build The API Image Locally

```bash
docker build \
  -f packages/aitraf-api/Dockerfile \
  -t aitraf-api:local \
  .
```

Expected outcome: the image builds without installing `aitraf-train` and without
copying `storage/`. The Dockerfile is expected at
`packages/aitraf-api/Dockerfile` and should copy repo `data/` into the image.

## Run A Runtime Smoke Check

Use real deployment-compatible model and storage inputs. The image includes repo
`data/`, so `AITRAF_DATA_PATH` may point at the copied image data path unless a
deployment mounts a replacement.

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

In another terminal:

```bash
curl http://localhost:8000/health
```

Expected outcome: the API returns a successful health response when model,
storage, and credential inputs are valid. Missing required environment or
artifacts should fail explicitly.

## Validate Workflow Design

Before merging implementation, inspect the workflow and confirm:

- Pushes to `master` and manual dispatch trigger the workflow.
- Train publish and API test/publish are separate paths.
- API publish depends on API tests.
- Train publish does not depend on API tests.
- API image publishes to
  `ghcr.io/${{ github.repository_owner }}/aitraf-api`.
- Train image continues to publish to
  `ghcr.io/${{ github.repository_owner }}/aitraf-train`.
