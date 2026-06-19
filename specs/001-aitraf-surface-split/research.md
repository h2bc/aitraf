# Research: AITRAF Surface Split

## Decision 1: Keep a single repository and split ownership with three first-class Python packages

- **Decision**: Implement the refactor as one monorepo with three explicit
  packages under `packages/`: `aitraf-core`, `aitraf-train`, and `aitraf-api`,
  each with its own `pyproject.toml` and `README.md`, while keeping one dev
  container and one shared workspace.
- **Rationale**: The product goal is not just code separation; it is clear
  package ownership for core/train/api. Package-local manifests and READMEs make
  dependency ownership and responsibilities explicit without splitting into
  multiple repositories.
- **Alternatives considered**:
  - Multiple repositories: rejected because it would break current workflow
    continuity and complicate shared config, storage, and experiment tracking.
  - Keep one `src/` tree with `aitraf_core`, `aitraf_train`, and `aitraf_api`:
    rejected because it would improve code organization but would not create
    first-class package manifests and docs for each surface.

## Decision 2: Move reusable runtime processing into `aitraf_core`

- **Decision**: Re-home clip/frame loading, pose extraction orchestration,
  VideoMAE feature generation, model-input preparation, and shared inference-ready
  adapters into `aitraf_core`.
- **Rationale**: The current code already shows shared seams in
  the reusable processing, data workflow, and model-processing helpers.
  These capabilities are needed by both offline training support flows and future
  API inference flows, so core should own them.
- **Clarification**: `data_ops` as a pipeline remains a train concern. What moves
  to core are the reusable primitives currently used by that pipeline, not the
  pipeline/orchestration surface itself. The implementation plan should preserve
  the existing repo shape here by keeping reusable code under `processing/` and
  selected shared helpers under `utils/`, rather than inventing a new top-level
  taxonomy.
- **Alternatives considered**:
  - Leave data ops in train and duplicate lightweight inference helpers in api:
    rejected because it would recreate the coupling this refactor is meant to remove.
  - Push all reusable logic into api-facing services: rejected because training
    and data preparation also need the same runtime capabilities without HTTP concerns.

## Decision 3: Keep training orchestration, Hydra configs, and experiment tracking in `aitraf-train`

- **Decision**: `aitraf-train` will own Hydra task dispatch, all training/offline
  `configs/`, prepare/train/eval/data-ops/label-ops scripts, task-model
  registrations, metrics/reporting, and MLflow-linked run behavior.
- **Rationale**: The existing `scripts/*.py`, task dispatch maps, config wiring,
  config wiring, and tracking modules are training-oriented control-plane code.
  Keeping them in train preserves operator workflows and keeps registry/run
  ownership out of core.
- **Clarification**: `data_ops` belongs here because it composes reusable core
  capabilities into an offline workflow. Core should expose clip/frame/pose/feature
  functions; train should decide when and how to run them.
- **Alternatives considered**:
  - Move command entrypoints into core: rejected because command orchestration is
    not a reusable runtime concern.
  - Share MLflow-facing code with api: rejected because the placeholder API should
    consume runtime capabilities, not training-run artifacts directly.

## Decision 4: Keep only thin repo-level convenience entrypoints; move actual configs and scripts into `aitraf-train`

- **Decision**: Move the actual Hydra config tree and offline scripts into
  `packages/aitraf-train`. Root `Taskfile.yml` owns global workspace tasks and
  includes package-owned task namespaces.
- **Rationale**: `configs/` is Hydra-specific and the current scripts are
  training/offline orchestration only. They belong to `aitraf-train`, not to
  `aitraf-core` or `aitraf-api`. Thin root wrappers are acceptable as ergonomics,
  but they are not the architectural owner.
- **Alternatives considered**:
  - Introduce a second command runner or new top-level CLI: rejected because it
    would add parallel workflow surfaces without user value.
  - Keep config and script ownership at repo root: rejected because it obscures
    the intended boundary that Hydra/offline orchestration belongs to train.

## Decision 5: Preserve existing internal module grouping where it already fits the boundary

- **Decision**: Avoid inventing new top-level folders inside core unless the
  current repo structure is clearly inadequate. Reuse the existing
  `processing/`, `processing/models/`, and selected shared `utils/` seams.
- **Rationale**: The current repository already separates reusable processing
  logic from workflow orchestration reasonably well. A large naming rewrite would
  add churn without improving the actual core/train/api boundary.
- **Alternatives considered**:
  - New top-level capability buckets such as `pose/`, `features/`, or
    `extractors/`: rejected because they describe an imagined architecture more
    than the actual codebase and would create unnecessary structural churn.

## Decision 6: Reserve an empty API package now

- **Decision**: Add `aitraf_api` as an empty package placeholder. It reserves
  future ownership for trick recognition and trick assessment, but this phase
  does not add routes, schemas, operations, adapters, smoke commands, or service
  runtime behavior.
- **Rationale**: The team is preparing for future inference, not delivering a
  service or contract now. An empty package makes ownership explicit without
  pretending that production serving or request schemas already exist.
- **Alternatives considered**:
  - Delay api artifacts entirely until inference work begins: rejected because the
    refactor needs to encode the future serving boundary now.
  - Add a partially working endpoint backed by ad hoc training code: rejected
    because it would create unreproducible behavior and confuse ownership.

## Decision 7: Separate reusable runtime artifacts from registry-managed artifacts

- **Decision**: `aitraf_core` may own deterministic runtime outputs such as clip
  frames, pose outputs, and VideoMAE feature tensors stored in documented repo
  storage, while `aitraf_train` owns run outputs, evaluation reports, and MLflow
  model registration concerns.
- **Rationale**: The spec explicitly distinguishes reusable processing from
  experiment-tracking and registry-managed assets. This boundary keeps core useful
  for both offline and online flows without embedding training-run semantics.
- **Alternatives considered**:
  - Let core write MLflow artifacts directly: rejected because it mixes reusable
    runtime logic with experiment orchestration.
  - Force all reusable outputs to stay in memory only: rejected because current
    data-prep and temporal-fusion workflows already rely on documented cached assets.

## Decision 8: Keep `data/`, `storage/`, and `notebooks/` as repo-level shared assets

- **Decision**: Keep `data/`, `storage/`, and `notebooks/` at the repository root
  instead of moving them under any single package.
- **Rationale**: These are workspace assets used across offline experimentation,
  shared runtime processing, and future serving preparation. They are not Python
  package surfaces and should not imply train-only or api-only ownership.
- **Alternatives considered**:
  - Move them under `aitraf-train`: rejected because that would incorrectly imply
    package ownership over shared assets and make future API preparation less clear.
  - Move them under `aitraf-core`: rejected because they are not reusable library code.

## Decision 9: Validate the split with command continuity, shared-core reuse, and empty API package checks

- **Decision**: Validation will cover three layers: existing prepare/train/eval
  command continuity, package-boundary checks added by this feature, and API
  placeholder checks that verify the package remains empty.
- **Rationale**: The refactor is architectural. Correctness depends more on stable
  boundaries and preserved workflows than on adding a new model result.
- **Alternatives considered**:
  - Validate only imports and package names: rejected because it would miss broken
    operator workflows and accidental API runtime additions.
  - Validate only full end-to-end training runs: rejected because that would be
    too slow and would not specifically prove shared-core reuse or API contract behavior.
