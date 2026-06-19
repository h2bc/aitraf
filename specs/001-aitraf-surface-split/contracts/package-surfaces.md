# Package Surface Contract

## Surface Names

The repository exposes three in-repo product surfaces:

- `aitraf-core`
- `aitraf-train`
- `aitraf-api`

Python import packages may use underscore names such as `aitraf_core`,
`aitraf_train`, and `aitraf_api`, but documentation and ownership references use
the hyphenated surface names above.

Each surface package must have:

- its own `pyproject.toml`
- its own `README.md`
- its own `src/<package_name>/` tree

## Dependency Rules

| Surface | May Depend On | Must Not Depend On | Owns |
|---------|---------------|--------------------|------|
| `aitraf-core` | External libraries, documented storage inputs | `aitraf-train`, `aitraf-api` | Shared runtime primitives, reusable prediction helpers, runtime asset loading/generation |
| `aitraf-train` | `aitraf-core`, external libraries, MLflow/Hydra/task configs | `aitraf-api` | Prepare/data_ops/train/eval orchestration, Hydra config ownership, task-model registrations, metrics, tracking |
| `aitraf-api` | `aitraf-core`, future serving libraries | `aitraf-train` | Empty reserved package for future API routes, request/response adapters, and serving behavior |

## Public Entry Contract

- `packages/aitraf-train/configs/` owns the Hydra config tree.
- `packages/aitraf-train/scripts/` owns the offline pipeline scripts.
- `data_ops` is a train-owned orchestration surface. It may call core helpers for
  frame extraction, pose inference, and feature generation, but those reusable
  helpers are not themselves a `data_ops` package inside core.
- `aitraf-core` should preserve the existing reusable structure where possible,
  especially `processing/`, `processing/models/`, and selected shared `utils/`
  modules, instead of introducing new top-level categories without a concrete
  repo-driven need.
- Repo-level Task includes expose train-owned commands under the `train:`
  namespace, such as `task train:prepare`, `task train:data_ops`,
  `task train:train`, `task train:eval`, and `task train:train_eval`.
- Shared clip/frame/pose/feature utilities must be imported from `aitraf-core`
  by training code now and future API-facing code where applicable.
- The API package is intentionally empty in this phase. Future API work is
  reserved for two operations: `trick-recognition` and `trick-assessment`.

## Ownership Rules

- Shared runtime outputs such as frame sampling, pose outputs, and reusable
  feature tensors belong to `aitraf-core`.
- Package-local dependency manifests must declare ownership where code directly
  uses a dependency. Shared runtime dependencies belong in `aitraf-core`,
  training/control-plane dependencies belong in `aitraf-train`, and API-serving
  dependencies belong in `aitraf-api`.
- MLflow logging, run directories, evaluation reports, and training-run metadata
  belong to `aitraf-train`.
- Future API contracts, request validation, and serving responses belong to
  `aitraf-api`.
- `data/`, `storage/`, and `notebooks/` remain repo-level shared assets rather
  than package-owned directories.

## Failure Behavior

- Missing surface registration, missing runtime assets, and invalid dependency
  direction must fail as explicit errors.
- API runtime invocation is unsupported because no API operations are implemented
  in this phase.
- Package boundary violations must be fixed at the owning package boundary.
