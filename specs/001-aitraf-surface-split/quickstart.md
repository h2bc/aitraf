# Quickstart: AITRAF Surface Split Validation

This guide describes the validation scenarios that must pass once the feature is
implemented.

## Prerequisites

- Dev container or Linux GPU host configured as described in [README.md](../../README.md)
- Repo workspace dependencies installed for all local packages
- Required environment variables and external credentials available for the
  existing data/training flows
- At least one representative clip available in repo-managed storage for smoke checks

## Scenario 1: Validate command continuity for training workflows

1. Install or refresh the workspace packages in editable mode.
2. Prepare an existing task manifest:

   ```bash
   task train:prepare -- task=trick_classification
   ```

3. Run a shared-processing smoke pass that exercises pose and VideoMAE feature generation:

   ```bash
   task train:data_ops -- pose_and_bbox_extraction.limit=1 video_mae_feature_extraction.limit=1
   ```

4. Run a representative training smoke command:

   ```bash
   task train:train -- task=trick_classification model=video_mae max_samples=8
   ```

5. Run a representative evaluation smoke command with a known model id:

   ```bash
   task train:eval -- task=score_prediction_ordinal model=video_mae model_id=<model-id>
   ```

Expected outcomes:

- The documented operator commands still work after the refactor, whether invoked
  directly from `packages/aitraf-train` or through thin repo-level wrappers.
- Training orchestration resolves through `aitraf-train`.
- Shared clip/frame/feature logic is consumed from `aitraf-core`.
- Missing registrations or assets fail explicitly with actionable errors.

## Scenario 2: Validate shared-core reuse in train-side workflows

1. Run lightweight package-boundary smoke commands:

   ```bash
   uv run python -c "import aitraf_core, aitraf_train, aitraf_api"
   uv run python -c "import importlib.util; assert importlib.util.find_spec('aitraf') is None"
   task --list
   ```

2. Inspect train-side extraction consumers and confirm shared processing imports
   resolve through `aitraf_core`.

Expected outcomes:

- `aitraf-core` has no reverse dependency on `aitraf-train` or `aitraf-api`.
- Train validation proves shared capabilities are defined once and reused.
- Import errors or dependency-direction violations fail either the targeted tests
  or the smoke validation added by this feature.

## Scenario 3: Validate empty API placeholder behavior

1. Inspect the reserved API package:

   ```bash
   find packages/aitraf-api/src/aitraf_api -maxdepth 2 -type f
   ```

Expected outcomes:

- The package contains only `__init__.py` in this implementation.
- No routes, schemas, operations, adapters, smoke commands, or service runtime are
  implemented yet.
- Future trick recognition and trick assessment work will compose
  `aitraf-core` and must not depend on `aitraf-train`.

## Scenario 4: Validate documentation and ownership clarity

1. Review the repository root documentation, each package `README.md`, and the
   package-surface contract.
2. Confirm that a maintainer can identify where to add:
   - Shared clip/pose/feature logic
   - Training-only orchestration or tracking changes
   - Future API-facing request handling

Expected outcomes:

- Ownership boundaries are clear within 10 minutes of repo inspection.
- Documentation points to the correct surface without conflicting terminology.
- Artifact ownership between shared runtime outputs and train-owned run outputs is explicit.
- Dependency ownership is explainable from the package manifests without relying
  on one giant undifferentiated environment.
