# Contract: API Docker Image

## Build Contract

The API image is built from the repository root with the API package Dockerfile.

```bash
docker build \
  --build-context aitraf_clips=storage/data/clips \
  -f packages/aitraf-api/Dockerfile \
  -t aitraf-api:local \
  .
```

## Required Image Contents

- `aitraf-api` package source and installed distribution
- `aitraf-core` package source and installed distribution
- Root `pyproject.toml` and `uv.lock` inputs used during build
- Repo `data/` directory for manifests and vocabularies
- Demo clips selected from the classification and AQA test manifests, copied
  from the `aitraf_clips` build context into `/workspace/storage/data/clips`
- Runtime system dependencies required by the API/core video path, including
  `ffmpeg`

## Prohibited Image Contents

- `packages/aitraf-train`
- Full `storage/`
- Local `.env` files or secrets
- Notebooks, generated runs, local models, local virtual environments, and cache
  directories

## Runtime Contract

The image starts the API server using the environment-backed app factory:

```text
aitraf_api.app:create_app_from_env
```

Required runtime environment:

- `AITRAF_API_TOKEN`
- `AITRAF_CLASSIFICATION_MODEL_URI`
- `AITRAF_AQA_MODEL_URI`
- `AITRAF_DATA_PATH`
- `AITRAF_STORAGE_PATH`
- `MLFLOW_TRACKING_URI`
- Any MLflow/artifact credentials required by the deployment

Expected runtime behavior:

- Valid runtime inputs allow the API process to start and expose `/health`.
- Missing runtime inputs fail explicitly.
- Missing model/storage artifacts fail explicitly.
- The image does not create or repair missing storage contents.
