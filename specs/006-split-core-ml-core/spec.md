# Feature Specification: Split Core And ML Core

**Feature Branch**: `[006-split-core-ml-core]`

**Created**: 2026-07-10

**Status**: Draft

**Input**: User description: "Refactor the existing core package into a lightweight `aitraf-core` package and a separate heavy `aitraf-ml-core` package so consumers can use basic shared helpers without installing the ML stack."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Use Lightweight Shared Helpers Independently (Priority: P1)

As a developer, I want to install and use basic shared AITRAF helpers without installing model, tensor, video-decoding, or experiment-tracking dependencies so that lightweight applications and tools remain small and focused.

**Why this priority**: The current package makes every consumer pay for the complete ML dependency stack even when it needs only dependency-light functionality. Removing that coupling is the primary purpose of the refactor.

**Independent Test**: Install only the lightweight core package in a clean environment, import and exercise every public lightweight helper, and confirm that no ML runtime package is required or installed.

**Acceptance Scenarios**:

1. **Given** a clean environment containing only `aitraf-core` and its declared dependencies, **When** a developer imports and uses a public lightweight helper, **Then** the helper works without any heavy ML runtime dependency being present.
2. **Given** a consumer that needs only lightweight helpers, **When** its dependency set is resolved, **Then** model execution, tensor processing, video decoding, and experiment-tracking libraries are absent from the resolved dependency graph.
3. **Given** a lightweight helper with invalid required input, **When** the developer invokes it, **Then** it raises an explicit error rather than importing an ML component or applying a fallback conversion.

---

### User Story 2 - Reuse ML Runtime Capabilities Explicitly (Priority: P2)

As a model engineer, I want reusable model loading, inference, preprocessing, and tensor or video processing to have a clearly named ML-specific owner so that training and offline inference workflows can depend on those capabilities intentionally.

**Why this priority**: Heavy reusable capabilities remain valuable, but their ownership and dependency cost must be explicit rather than hidden behind a general-purpose core package.

**Independent Test**: Install the ML core package with its declared dependencies and run representative existing model loading, preprocessing, processing, and inference tests through its documented public surface.

**Acceptance Scenarios**:

1. **Given** an existing reusable ML runtime capability, **When** a maintainer locates it after the refactor, **Then** it is owned by `aitraf-ml-core` and has no duplicate implementation in `aitraf-core` or `aitraf-train`.
2. **Given** a training or offline workflow that uses reusable ML runtime behavior, **When** the workflow runs after migration, **Then** it obtains that behavior from `aitraf-ml-core` through the new required import surface.
3. **Given** a missing or incompatible ML dependency or artifact, **When** an ML core capability is invoked, **Then** the operation fails with actionable context and does not fall back to an alternate implementation.

---

### User Story 3 - Maintain Clear Dependency Direction (Priority: P3)

As a repository maintainer, I want package ownership and dependency direction to be documented and automatically checked so that future changes do not reintroduce heavy dependencies into the lightweight core.

**Why this priority**: The immediate move is insufficient unless the lightweight boundary remains enforceable as the repository evolves.

**Independent Test**: Review the documented package rules and run automated boundary checks that fail when lightweight core code imports ML core or any prohibited heavy dependency.

**Acceptance Scenarios**:

1. **Given** a proposed change to `aitraf-core`, **When** it imports `aitraf-ml-core` or a prohibited heavy dependency, **Then** repository validation fails.
2. **Given** a new reusable helper, **When** a maintainer consults the package ownership documentation, **Then** they can determine whether it belongs in lightweight core, ML core, train, or the API based on its consumers and dependency needs.
3. **Given** the completed migration, **When** a maintainer searches for the previous ML modules under `aitraf-core`, **Then** no compatibility aliases, forwarding modules, or duplicate implementations remain.

### Edge Cases

- A currently lightweight helper imports a heavy module indirectly through a package initializer or convenience re-export.
- A module combines dependency-free transformations with tensor-aware behavior and therefore cannot move intact to either boundary.
- An ML runtime module needs a lightweight type or helper, creating a valid dependency from ML core to lightweight core.
- A lightweight helper currently exposes a heavy-library type in its public input or output contract.
- Existing callers import from package-level aggregators whose exports would accidentally load the full ML runtime.
- Required model artifacts, storage settings, or runtime dependencies are absent after the package migration.
- A third-party dependency classified as lightweight pulls in a prohibited ML dependency transitively.
- Old module paths remain importable accidentally because stale files, editable installations, or compatibility re-exports were retained.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The repository MUST expose a lightweight shared package named `aitraf-core` and a heavy reusable ML runtime package named `aitraf-ml-core`.
- **FR-002**: `aitraf-core` MUST own only shared capabilities that do not require model execution, tensor processing, video decoding, model hubs, or experiment-tracking runtimes.
- **FR-003**: `aitraf-ml-core` MUST own reusable model loading, inference, preprocessing, tensor processing, and video processing currently owned by `aitraf-core` when those capabilities serve more than one ML workflow or task.
- **FR-004**: Feature-specific offline data, training, evaluation, metrics, tracking orchestration, configurations, and scripts MUST remain owned by `aitraf-train`.
- **FR-005**: Serving-only behavior MUST remain owned by `aitraf-api`; neither core package may absorb API routes, startup state, authentication, or response composition.
- **FR-005a**: API thumbnail generation and media URL behavior MUST use the shared `aitraf-core` S3 client, URI parsing, object-existence, and presigning helpers instead of maintaining API-local equivalents.
- **FR-006**: The package dependency direction MUST permit `aitraf-ml-core` to depend on `aitraf-core` while prohibiting `aitraf-core` from depending on `aitraf-ml-core`.
- **FR-007**: Installing `aitraf-core` alone MUST NOT install or require Torch, TorchCodec, Transformers, MLflow, Hugging Face Hub, Kornia, PyAV, or equivalent model/tensor/video runtime dependencies; Boto3 is permitted for the shared S3 surface.
- **FR-008**: Importing any documented public `aitraf-core` module MUST NOT import an ML runtime module directly or indirectly.
- **FR-009**: Existing in-repository consumers MUST be updated to the new package ownership and import paths in the same change.
- **FR-010**: The refactor MUST remove obsolete ML modules from `aitraf-core` and MUST NOT provide forwarding modules, aliases, optional legacy paths, or compatibility re-exports for their former locations.
- **FR-011**: Modules that mix lightweight and ML-specific behavior MUST be decomposed so each resulting capability has a single appropriate owner and an explicit typed boundary.
- **FR-012**: Public package initializers MUST expose only symbols owned by that package and MUST NOT cause unrelated optional surfaces to load as an import side effect.
- **FR-013**: Each moved capability MUST retain its required input and output contract unless the migration documents and updates all affected in-repository callers to one new contract.
- **FR-014**: Missing dependencies, required configuration, model artifacts, or invalid inputs MUST produce explicit errors at the owning boundary without silent defaults, alternate implementations, or input-shape normalization.
- **FR-015**: Repository documentation MUST define the responsibility, allowed dependency categories, public surface, and consumer relationship for both core packages.
- **FR-016**: The refactor MUST classify every existing `aitraf-core` module as retained in `aitraf-core`, moved to `aitraf-ml-core`, moved to a feature-owning package, decomposed, or removed; no module may remain unclassified.

### Architecture And Reuse Requirements *(mandatory)*

- **AR-001**: The feature MUST change the package metadata and source surfaces for `packages/aitraf-core` and the new `packages/aitraf-ml-core`, plus affected callers in `packages/aitraf-train` and the API thumbnail/media S3 integration in `packages/aitraf-api`.
- **AR-002**: The new `aitraf-ml-core` surface is justified because reusable heavy ML runtime behavior has materially different consumers and dependency weight from lightweight shared helpers and must be independently installable.
- **AR-003**: Shared logic MUST have one owner. Capabilities MUST NOT be copied between core, ML core, train, or API to complete the migration.
- **AR-004**: Production behavior MUST remain in versioned package modules and documented command surfaces rather than notebooks or local ad hoc scripts.
- **AR-005**: Decomposed helpers MUST prefer pure transformations with explicit inputs and outputs; model, cache, filesystem, network, and framework state MUST remain localized at their owning boundaries.
- **AR-006**: All callers, tests, package metadata, development commands, container definitions, and documentation MUST adopt the new package boundary directly, and superseded paths and dependency declarations MUST be removed.
- **AR-007**: Each changed boundary MUST accept one required input representation and return one documented output representation; alternate shapes and compatibility coercions MUST be rejected.

### Validation And Reproducibility Requirements *(mandatory)*

- **VR-001**: Validation MUST include isolated installation and import tests for `aitraf-core`, isolated package tests for `aitraf-ml-core`, affected training tests, and command-level smoke validation of at least one representative existing ML workflow.
- **VR-002**: Verification MUST record the package manifests, resolved dependency sets, test commands, representative workflow configuration, model reference, artifact assumptions, and input fixture used to reproduce the result.
- **VR-003**: Validation MUST demonstrate explicit failure for at least one missing ML artifact or dependency and at least one invalid lightweight-core input, with no silent fallback behavior.
- **VR-004**: Because this feature changes ownership rather than model behavior, representative pre-refactor and post-refactor workflow outputs MUST be equivalent for the same versioned inputs and configuration; no new evaluation metric or artifact format is required.
- **VR-005**: Automated architecture validation MUST detect prohibited direct and transitive heavy dependencies in `aitraf-core` and any import of `aitraf-ml-core` by `aitraf-core`.
- **VR-006**: Validation MUST confirm that every prior in-repository import of a moved `aitraf-core` capability is either updated to its new owner or removed.

### Key Entities *(include if feature involves data)*

- **Lightweight Core Capability**: A shared, dependency-light function or type with explicit inputs and outputs that can be used without installing the ML runtime stack.
- **ML Core Capability**: Reusable model loading, inference, preprocessing, tensor processing, or video processing behavior whose dependency cost is intentionally isolated from lightweight consumers.
- **Package Boundary Rule**: A verifiable constraint defining which capabilities and dependency categories a package may own and which dependency directions are permitted.
- **Module Classification**: The migration decision for each existing core module: retain, move to ML core, move to a feature owner, decompose, or remove.

## Architecture And Data Impact

- **Touched Surfaces**: `packages/aitraf-core`, new `packages/aitraf-ml-core`, affected `packages/aitraf-train` modules and package metadata, API thumbnail and media URL S3 helpers, root dependency and development configuration, tests, container or task definitions that install core packages, and architecture documentation.
- **Shared Helpers To Add Or Extend**: Only dependency-light capabilities proven to have multiple consumers remain in `aitraf-core`; reusable model loading, inference, preprocessing, sampling that requires tensor types, tensor processing, and video decoding move to `aitraf-ml-core`. Single-consumer helpers may instead move to their feature-owning package.
- **Legacy Surfaces Removed**: Former ML module paths beneath `aitraf-core`, stale package exports, obsolete dependency declarations, duplicate implementations, forwarding imports, and compatibility aliases.
- **Required Input Shapes**: Existing typed contracts remain the required shapes unless explicitly replaced for all in-repository callers. Package boundaries may not expose an undeclared alternate scalar, collection, mapping, path, serialized, tensor, or model representation.
- **Data Or Artifact Impact**: No dataset, label vocabulary, prediction schema, metric definition, or tracked artifact format is intentionally changed. Cache and artifact locations remain unchanged unless a later plan identifies a package-name-derived path that must change and documents its direct migration.
- **Reproducibility Inputs**: Package manifests and resolved dependency reports, existing workflow configurations and seeds, representative input fixtures, required model and artifact references, documented test and smoke commands, and before/after outputs for the chosen representative workflow.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In a clean environment, 100% of documented `aitraf-core` imports and tests succeed while zero prohibited ML runtime packages are installed.
- **SC-002**: 100% of existing core modules receive an explicit retain, move, decompose, feature-localize, or remove decision, and no obsolete ML import path remains usable from the repository source tree.
- **SC-003**: 100% of affected in-repository tests pass using the new package ownership, and at least one representative ML workflow completes with equivalent outputs for the same inputs and configuration.
- **SC-004**: Automated dependency-boundary checks reject every tested attempt to add a prohibited heavy dependency or ML-core import to `aitraf-core`.
- **SC-005**: A maintainer can determine the correct owner for a proposed lightweight helper or ML runtime capability within 5 minutes using repository documentation alone.
- **SC-006**: The dependency report for an `aitraf-core`-only consumer contains zero model execution, tensor processing, video decoding, model hub, or experiment-tracking runtime packages.

## Assumptions

- `aitraf-core` remains the stable name for lightweight shared capabilities; `aitraf-ml-core` becomes the stable name for reusable heavy ML runtime capabilities.
- `aitraf-train` consumes both packages, while `aitraf-api` consumes `aitraf-core` for shared S3 behavior and never consumes `aitraf-ml-core`.
- Shared S3 client, URI, existence, and presigning behavior remains in core because API and train both consume it; clip-download orchestration remains train-owned.
- Optional dependency extras are not the target architecture because they do not provide an enforceable ownership boundary between lightweight and ML runtime code.
- The change is a direct migration with no legacy import compatibility period.
- Model behavior, datasets, labels, metrics, and artifact formats remain unchanged unless a required package-boundary correction is explicitly documented during planning.
- Invalid inputs and missing runtime requirements fail explicitly rather than being repaired, guessed, or routed through an alternate implementation.
