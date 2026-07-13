# AITRAF: Aggressive Inline Trick Recognition and Feedback

AITRAF is a monorepo for inline skating trick recognition and performance
feedback.

## Packages

- `packages/aitraf-core`: lightweight shared cache, structured-file, and S3 helpers.
- `packages/aitraf-ml-core`: reusable model loading, inference, preprocessing, tensor, and video runtime.
- `packages/aitraf-train`: Hydra-driven data ops, label ops, preparation,
  training, evaluation, metrics, tracking, configs, and scripts.
- `packages/aitraf-api`: FastAPI inference service for demo video listing,
  trick classification, and trick AQA predictions.

## Workspace Setup

1. Start the dev container.
2. Copy `.env.example` to `.env` and fill in required AWS, MLflow, data path,
   storage path, API token, and registered model URI values.
3. Install workspace dependencies:

```bash
task install
```

Run the local API and Redis as sibling containers managed by Docker Compose from
the devcontainer:

```bash
task api:run
```

Press `Ctrl+C` to stop both services. `docker compose down` removes the local
containers and network while preserving the visitor-count volume. The local API
is available at `http://localhost:8001` by default.

## Workspace Commands

Root commands are workspace-level only:

```bash
task install
task lint
task format
```

Use `task lint` and `task format` for workspace-wide checks from the repo root.

Train workflows are exposed through the `train:` task namespace. See
[packages/aitraf-train/README.md](packages/aitraf-train/README.md) for commands
and workflow documentation.

API workflows are exposed through the `api:` task namespace:

```bash
task api:run
task api:test
```

`task api:run` builds and starts both the API and Redis through local Compose.
It runs in the foreground, so `Ctrl+C` stops both services natively while
preserving the Redis named volume and visitor count. API source is mounted into
the container and reloads on changes.

See [packages/aitraf-api/README.md](packages/aitraf-api/README.md) for runtime
environment requirements and endpoint behavior.

## Shared Workspace Assets

- `data/`: lightweight manifests and annotation-derived inputs.
- `storage/`: generated caches, model assets, run outputs, and larger local
  artifacts.
- `notebooks/`: analysis notebooks.

## Package Docs

- [aitraf-core](packages/aitraf-core/README.md)
- [aitraf-ml-core](packages/aitraf-ml-core/README.md)
- [aitraf-train](packages/aitraf-train/README.md)
- [aitraf-api](packages/aitraf-api/README.md)
