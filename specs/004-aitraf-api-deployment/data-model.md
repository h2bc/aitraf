# Data Model: AITRAF API Deployment

## API Dockerfile

Represents the versioned build definition for the API serving image.

**Fields**:

- `path`: `packages/aitraf-api/Dockerfile`
- `base_runtime`: CUDA runtime image aligned with the existing train Dockerfile
  unless implementation finds a tested smaller runtime that still satisfies
  locked API/core dependencies
- `package_target`: `aitraf-api`
- `included_packages`: `packages/aitraf-api`, `packages/aitraf-core`
- `included_repo_data`: `data/`
- `excluded_surfaces`: `packages/aitraf-train`, `storage/`, notebooks, generated
  runs, local models, secrets, local virtual environments
- `entrypoint`: API server factory entrypoint for `aitraf_api.app:create_app_from_env`
- `dependency_install`: frozen, non-dev, non-editable package-scoped workspace
  install

**Validation rules**:

- Must build from the repository root context.
- Must not install `aitraf-train`.
- Must not copy `storage/` or committed secrets.
- Must fail during build if required package metadata or lockfile inputs are
  missing.

## API Deployment Image

Represents the built and published API serving artifact.

**Fields**:

- `image_name`: `ghcr.io/<owner>/aitraf-api`
- `tags`: `latest`, short source revision
- `labels`: source repository and source revision metadata from the workflow
- `runtime_data_path`: path to copied repo `data/` or an explicitly supplied
  replacement data path
- `runtime_storage_path`: externally mounted or externally available storage path
- `required_env`: `AITRAF_API_TOKEN`, `AITRAF_CLASSIFICATION_MODEL_URI`,
  `AITRAF_AQA_MODEL_URI`, `AITRAF_DATA_PATH`, `AITRAF_STORAGE_PATH`,
  `MLFLOW_TRACKING_URI`, plus any MLflow credentials required by the deployment

**Validation rules**:

- Must start the API process with explicit runtime environment.
- Must expose the health route when required model/storage/credential inputs are
  valid.
- Must fail explicitly when required environment or artifacts are missing.

## GitHub Image Publishing Workflow

Represents the repository automation for publishing Docker images.

**Fields**:

- `path`: `.github/workflows/publish-docker-images.yml`
- `triggers`: push to `master`, manual dispatch
- `registry`: `ghcr.io`
- `train_image`: `ghcr.io/<owner>/aitraf-train`
- `api_image`: `ghcr.io/<owner>/aitraf-api`
- `train_job`: builds and publishes the existing train image
- `api_test_job`: runs API tests before API publish
- `api_publish_job`: builds and publishes the API image only after API tests pass
- `permissions`: read repository contents and write packages where publishing is
  performed

**Validation rules**:

- Train publishing must not depend on API tests.
- API publishing must depend on API tests.
- API image publishing must not run when API tests fail.
- Both images must use stable latest and source-revision-specific references.

## Runtime Configuration

Represents environment-specific inputs supplied outside committed files.

**Fields**:

- `api_token`: bearer token expected from frontend clients
- `classification_model_uri`: MLflow model URI for trick classification
- `aqa_model_uri`: MLflow model URI for temporal-fusion trick AQA
- `data_path`: path to manifests and vocabularies, normally the copied image
  `data/` directory
- `storage_path`: path to clips, feature caches, and model cache inputs
- `tracking_uri`: MLflow tracking URI
- `tracking_credentials`: deployment-provided credentials required by MLflow or
  artifact storage

**Validation rules**:

- Secrets and credentials must not be committed into the Dockerfile or workflow.
- Missing values must fail explicitly through existing API configuration loading
  or runtime readiness behavior.
- `storage_path` must remain external to the image.
