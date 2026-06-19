# AITRAF: Aggressive Inline Trick Recognition and Feedback

AITRAF is a monorepo for inline skating trick recognition and performance
feedback.

## Packages

- `packages/aitraf-core`: shared runtime processing and model-input helpers.
- `packages/aitraf-train`: Hydra-driven data ops, label ops, preparation,
  training, evaluation, metrics, tracking, configs, and scripts.
- `packages/aitraf-api`: empty package reserved for future inference API work.

## Workspace Setup

1. Start the dev container.
2. Copy `.env.example` to `.env` and fill in required AWS and MLflow values.
3. Install workspace dependencies:

```bash
uv sync
```

## Workspace Commands

Root commands are workspace-level only:

```bash
task lint
task lint -- --fix
task format
task format -- --check
```

Train workflows are exposed through the `train:` task namespace. See
[packages/aitraf-train/README.md](packages/aitraf-train/README.md) for commands
and workflow documentation.

## Shared Workspace Assets

- `data/`: lightweight manifests and annotation-derived inputs.
- `storage/`: generated caches, model assets, run outputs, and larger local
  artifacts.
- `notebooks/`: analysis notebooks.

## Package Docs

- [aitraf-core](packages/aitraf-core/README.md)
- [aitraf-train](packages/aitraf-train/README.md)
- [aitraf-api](packages/aitraf-api/README.md)
