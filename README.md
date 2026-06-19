# AITRAF: Aggressive Inline Trick Recognition and Feedback

AITRAF is a monorepo for inline skating trick recognition and performance
feedback.

## Packages

- `packages/aitraf-core`: shared runtime processing and model-input helpers.
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

See [packages/aitraf-api/README.md](packages/aitraf-api/README.md) for runtime
environment requirements and endpoint behavior.

## Shared Workspace Assets

- `data/`: lightweight manifests and annotation-derived inputs.
- `storage/`: generated caches, model assets, run outputs, and larger local
  artifacts.
- `notebooks/`: analysis notebooks.

## Package Docs

- [aitraf-core](packages/aitraf-core/README.md)
- [aitraf-train](packages/aitraf-train/README.md)
- [aitraf-api](packages/aitraf-api/README.md)
