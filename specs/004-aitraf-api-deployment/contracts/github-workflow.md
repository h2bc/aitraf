# Contract: GitHub Image Publishing Workflow

## Workflow Scope

The existing Docker image publishing workflow is extended to publish both train
and API images.

```text
.github/workflows/publish-docker-images.yml
```

## Triggers

- Push to `master`
- Manual workflow dispatch

## Images

- Train: `ghcr.io/<repository-owner>/aitraf-train`
- API: `ghcr.io/<repository-owner>/aitraf-api`

## Tags And Labels

Each published image must include:

- `latest`
- short source-revision tag
- image labels generated from repository metadata

## Job Contract

### Train Publish Job

- Checks out the repository.
- Sets up Docker Buildx.
- Logs in to GHCR.
- Builds and publishes `packages/aitraf-train/Dockerfile`.
- Does not depend on API tests.

### API Test Job

- Checks out the repository.
- Sets up the Python workspace with locked dependencies.
- Runs `task api:test` or equivalent `uv run pytest packages/aitraf-api/tests`.
- Fails the API path when tests fail.

### API Publish Job

- Depends on the API test job.
- Checks out the repository.
- Sets up Docker Buildx.
- Logs in to GHCR.
- Builds and publishes `packages/aitraf-api/Dockerfile`.
- Publishes only after API tests pass.

## Failure Contract

- API test failure prevents API image publishing.
- API test failure does not prevent the train publish job from continuing.
- Train build/publish failure does not publish a successful train candidate.
- API build/publish failure does not publish a successful API candidate.
- Workflow permissions must include package write access for publish jobs.
