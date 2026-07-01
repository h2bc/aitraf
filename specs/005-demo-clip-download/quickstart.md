# Quickstart: Demo Clip Download

## Prerequisites

- Repository dependencies installed through `uv`
- API manifests available under `data/manifests`
- Object storage credentials and endpoint settings for runtime download smoke
  validation:
  - `AWS_ENDPOINT_URL`
  - `AWS_DEFAULT_REGION`
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
- API runtime settings for startup smoke validation:
  - `AITRAF_API_TOKEN`
  - `AITRAF_CLASSIFICATION_MODEL_URI`
  - `AITRAF_AQA_MODEL_URI`
  - `AITRAF_API_DEVICE`
  - `AITRAF_DATA_PATH`
  - `AITRAF_STORAGE_PATH`
  - `MLFLOW_TRACKING_URI`

## Validate Shared Downloader Tests

Run package tests for shared clip download behavior:

```bash
uv run --package aitraf-core pytest packages/aitraf-core/tests
```

Expected outcome: core tests pass, including object storage URI parsing,
skip-existing behavior, force behavior, and download error propagation.

## Validate API Demo Download Tests

Run API tests:

```bash
uv run --package aitraf-api --extra dev pytest packages/aitraf-api/tests
```

Expected outcome: API tests pass, including demo clip selection and runtime
download request construction.

## Validate Train Data Ops Reuse

Run or test the train data-op path that downloads clips:

```bash
task train:data_ops -- download_clips.enabled=true
```

Expected outcome: train labels still produce the same clip destinations, existing
clips are skipped unless force is enabled, and failed source downloads are
reported explicitly.

Validation note from 2026-07-01: the full `task train:data_ops` command was not
run because it executes the broader configured data pipeline. The train-owned
labels-to-download reuse path was validated with:

```bash
uv run --package aitraf-train pytest packages/aitraf-train/tests
```

Result: `3 passed`.

## Validate API Image Builds Without Clips

Build the API image without providing local clips:

```bash
docker build -f packages/aitraf-api/Dockerfile -t aitraf-api:local .
```

Expected outcome: the image builds without `storage/data/clips`, without
downloading clips, and without requiring an `aitraf_clips` build context.

## Runtime Demo Download Smoke

Use a temporary writable storage directory and enable runtime demo clip download
explicitly:

```bash
docker run --rm -p 8000:8000 \
  --env-file .env \
  -e AITRAF_API_TOKEN="$AITRAF_API_TOKEN" \
  -e AITRAF_CLASSIFICATION_MODEL_URI="$AITRAF_CLASSIFICATION_MODEL_URI" \
  -e AITRAF_AQA_MODEL_URI="$AITRAF_AQA_MODEL_URI" \
  -e AITRAF_API_DEVICE="$AITRAF_API_DEVICE" \
  -e AITRAF_DATA_PATH=/workspace/data \
  -e AITRAF_STORAGE_PATH=/workspace/storage \
  -e AITRAF_API_DEMO_CLIPS_DOWNLOAD=1 \
  -e AWS_ENDPOINT_URL="$AWS_ENDPOINT_URL" \
  -e AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
  -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
  --tmpfs /workspace/storage:rw,size=512m \
  aitraf-api:local
```

Expected outcome: selected demo clips are present under
`/workspace/storage/data/clips` in the container before demo inference is
considered ready.
If `.env` already contains the required API, model, MLflow, and object storage
settings, keep `--env-file .env` and only override container-specific paths with
the explicit `-e AITRAF_DATA_PATH` and `-e AITRAF_STORAGE_PATH` values.

Validation note from 2026-07-01: the image build passed with:

```bash
docker build -f packages/aitraf-api/Dockerfile -t aitraf-api:local .
```

Result: Docker completed the build and tagged `aitraf-api:local` without an
`aitraf_clips` build context.

The runtime hydration smoke was validated without model loading by running the
built image with `--env-file .env`, `AITRAF_API_DEMO_CLIPS_DOWNLOAD=1`, and
`--tmpfs /workspace/storage:rw,size=512m`, then invoking
`hydrate_demo_clips(settings)` inside the container. Result:
`downloaded_or_present_files=10`.

## Failure Smoke

Run the same startup with invalid object storage credentials or an unwritable
storage mount.

Expected outcome: startup fails explicitly and identifies the missing credential,
unavailable source, or unwritable destination rather than silently serving a
partial demo set.
