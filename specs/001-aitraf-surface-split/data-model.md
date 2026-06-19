# Data Model: AITRAF Surface Split

## Entity: Repository Surface

- **Description**: A named in-repo product boundary with an explicit ownership role.
- **Fields**:
  - `surface_name`: `aitraf-core`, `aitraf-train`, or `aitraf-api`
  - `import_package`: Python import package associated with the surface
  - `package_root`: Package directory under `packages/`
  - `package_manifest`: Package-local `pyproject.toml`
  - `package_readme`: Package-local `README.md`
  - `responsibilities`: List of owned behaviors
  - `allowed_dependencies`: Other surfaces this surface may depend on
  - `public_entrypoints`: Commands, modules, or contracts exposed to developers
- **Relationships**:
  - `aitraf-train` depends on `aitraf-core`
  - `aitraf-api` depends on `aitraf-core`
  - `aitraf-core` depends on no surface-local consumers
- **Validation Rules**:
  - Surface names are fixed to the three approved names
  - `aitraf-core` must not import from `aitraf-train` or `aitraf-api`
  - `aitraf-train` must not depend on `aitraf-api`
  - Each surface must have its own package manifest and README
  - Existing reusable module groupings should be preserved unless the plan
    explicitly justifies a structural rewrite
  - Each repository-owned module belongs to exactly one surface
- **State Transitions**:
  - `planned` -> `implemented` -> `validated`

## Entity: Shared Processing Capability

- **Description**: A reusable runtime capability that transforms clip inputs into
  frames, pose outputs, feature outputs, or other prediction-ready representations.
- **Fields**:
  - `capability_name`
  - `input_type`
  - `output_type`
  - `runtime_dependencies`
  - `artifact_behavior`
  - `failure_modes`
- **Relationships**:
  - Owned by `Repository Surface = aitraf-core`
  - Consumed by `Training Workflow Adapter` and `Inference Operation Contract`
- **Validation Rules**:
  - Must have explicit input and output contracts
  - Should remain located under existing reusable surfaces such as `processing/`
    or selected shared helpers unless a clearer repo-backed structure is needed
  - Must declare whether outputs are persisted in documented storage or returned in-memory
  - Must fail explicitly on missing clips, assets, or incompatible configuration
- **State Transitions**:
  - `declared` -> `wired` -> `shared-by-train-and-api`

## Entity: Runtime Asset Reference

- **Description**: A reference to reusable inputs or outputs needed by shared
  runtime logic or training orchestration.
- **Fields**:
  - `asset_kind`: clip, frame batch, pose output, feature tensor, model weights, manifest, or run artifact
  - `owner_surface`
  - `storage_location`
  - `version_or_identity`
  - `creation_mode`: generated, cached, downloaded, or registered
  - `consumers`
- **Relationships**:
  - Produced by `Shared Processing Capability` or `Training Workflow Adapter`
  - Referenced by `Inference Operation Contract`
- **Validation Rules**:
  - Storage location must map to documented repo storage or an approved external registry
  - Owner surface must be unambiguous
  - Registry-managed assets cannot be silently substituted for missing runtime assets
- **State Transitions**:
  - `missing` -> `available` -> `consumed` -> `stale` or `replaced`

## Entity: Repo Shared Asset

- **Description**: A repo-level non-package directory used across multiple surfaces.
- **Fields**:
  - `asset_root`: `data/`, `storage/`, or `notebooks/`
  - `purpose`
  - `consuming_surfaces`
  - `ownership_policy`
- **Relationships**:
  - Referenced by `Runtime Asset Reference`
  - Consumed by `Repository Surface`
- **Validation Rules**:
  - Repo shared assets must not be relocated under a single package unless the
    plan explicitly changes cross-surface ownership
  - Documentation must state why the asset stays repo-level
- **State Transitions**:
  - `documented` -> `used` -> `reorganized`

## Entity: Training Workflow Adapter

- **Description**: A train-surface orchestration path that binds Hydra configs,
  manifests, task/model dispatch, metrics, and tracking to shared core capabilities.
- **Fields**:
  - `workflow_name`: prepare, data_ops, train, eval, or train_eval
  - `task_name`
  - `model_name`
  - `config_inputs` from `packages/aitraf-train/configs/`
  - `shared_capabilities_used`
  - `tracking_outputs`
- **Relationships**:
  - Owned by `Repository Surface = aitraf-train`
  - Uses `Shared Processing Capability`
  - Produces `Runtime Asset Reference` values and MLflow/run artifacts
- **Validation Rules**:
  - Must keep documented command entrypoints valid
  - Must not duplicate core-owned runtime transformations
  - Must own Hydra config and offline script surfaces
  - Must surface missing registrations or unsupported task/model combinations as errors
- **State Transitions**:
  - `configured` -> `dispatched` -> `completed` or `failed`

## Entity: Future Inference Operation

- **Description**: A future externally callable operation reserved for the API surface, not implemented in this phase.
- **Fields**:
  - `operation_name`: `trick-recognition` or `trick-assessment`
  - `request_schema`
  - `success_schema`
  - `implementation_status`: future
  - `required_core_capabilities`
  - `required_assets`
- **Relationships**:
  - Owned by `Repository Surface = aitraf-api`
  - Depends on `Shared Processing Capability`
  - Reads `Runtime Asset Reference` values for models and clip inputs
- **Validation Rules**:
  - No operation modules, schemas, adapters, routes, or smoke commands exist in this phase
  - Future operation implementation must compose `aitraf-core`
  - Future operation implementation must not depend on `aitraf-train`
- **State Transitions**:
  - `reserved` -> `enabled` -> `validated`

## Entity: Artifact Ownership Boundary

- **Description**: The policy that determines whether an output belongs to shared
  runtime processing or to train-owned experiment tracking and registry workflows.
- **Fields**:
  - `boundary_name`
  - `owned_by`
  - `covered_artifact_types`
  - `excluded_artifact_types`
  - `documentation_reference`
- **Relationships**:
  - Applied to `Shared Processing Capability`, `Runtime Asset Reference`, and `Training Workflow Adapter`
- **Validation Rules**:
  - Shared runtime outputs cannot write MLflow-specific state directly
  - Train-owned tracking outputs cannot be required for future API invocation unless explicitly documented
  - Package dependency ownership must be inferable from package manifests
  - Every moved output path must have one documented owner
- **State Transitions**:
  - `defined` -> `enforced` -> `documented`
