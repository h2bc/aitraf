# Feature Specification: AITRAF Surface Split

**Feature Branch**: `001-aitraf-surface-split`

**Created**: 2026-06-19

**Status**: Draft

**Input**: User description: "Refactor the repository from a single training-focused package into `aitraf-core`, `aitraf-train`, and `aitraf-api` within one repo, with shared reusable logic in core and a placeholder API prepared for future inference."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Separate Shared And Training Responsibilities (Priority: P1)

As a repository maintainer, I want the repo to clearly separate shared reusable capabilities from training-only orchestration so that future product work can build on the same foundation without duplicating logic.

**Why this priority**: This is the core structural change. Without it, future API work would either duplicate training logic or continue coupling reusable processing to experiment-specific workflows.

**Independent Test**: Review the refactored repository surfaces and confirm that shared processing and prediction capabilities have one documented owner, while current training workflows still have a dedicated training surface.

**Acceptance Scenarios**:

1. **Given** the current training-focused repository, **When** a maintainer reviews the refactored structure, **Then** the repo exposes distinct `aitraf-core`, `aitraf-train`, and `aitraf-api` surfaces with documented responsibilities.
2. **Given** a capability needed by both offline workflows and future serving, **When** the maintainer traces its ownership, **Then** that capability is defined in `aitraf-core` rather than duplicated across other surfaces.

---

### User Story 2 - Reuse Shared Processing Across Offline And Online Flows (Priority: P2)

As a model engineer, I want shared clip processing, reusable prediction inputs, and common feature-generation paths to be available through one core surface so that batch preparation, training support workflows, and future request-time inference stay aligned.

**Why this priority**: The refactor is valuable only if it removes duplicated processing paths and keeps offline and future online behavior consistent for the same inputs.

**Independent Test**: Confirm that the shared core surface defines reusable contracts for clip-to-frame processing, shared prediction-ready outputs, and other non-registry processing needed by both training and serving flows.

**Acceptance Scenarios**:

1. **Given** a workflow that needs clip-derived reusable outputs, **When** an engineer integrates it into training support logic or serving logic, **Then** the same core-owned processing path can be invoked in both contexts.
2. **Given** a shared processing capability that does not belong in experiment tracking, **When** it is added to the refactored repo, **Then** it is owned by the core surface and not by a training-only or serving-only wrapper.

---

### User Story 3 - Reserve A Future Inference Surface (Priority: P3)

As an API developer, I want an empty reserved `aitraf-api` surface for future trick recognition and trick assessment work so that inference code has an explicit future home without adding runtime API behavior now.

**Why this priority**: The API is not being delivered now, but reserving the package reduces future ambiguity and forces clear boundaries during the refactor.

**Independent Test**: Review the placeholder API package and documentation to confirm it is intentionally empty except for package metadata/documentation and contains no routes, schemas, operations, adapters, or app runtime.

**Acceptance Scenarios**:

1. **Given** a developer starting future inference work, **When** they inspect the repository, **Then** they can identify `aitraf-api` as the reserved package for future trick recognition and trick assessment work.
2. **Given** the current refactor phase, **When** a maintainer inspects `aitraf-api`, **Then** the package contains no runtime API implementation beyond the package initializer.

### Edge Cases

- A capability appears to be shared, but currently exists only inside a training-specific workflow with no reusable contract.
- Current task names and future API operation names refer to the same business outcome with different terminology, creating ownership ambiguity.
- Shared prediction inputs depend on required model assets, manifests, or configuration that are missing at runtime.
- A core processing step starts writing experiment-tracking or registry state even though it is intended to remain a reusable runtime capability.
- API runtime modules are accidentally added even though inference implementation is out of scope for this phase.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The repository MUST expose three distinct in-repo product surfaces named `aitraf-core`, `aitraf-train`, and `aitraf-api`.
- **FR-002**: The system MUST define and document the responsibility boundary for each surface, including which surface owns shared processing, training orchestration, and future inference entrypoints.
- **FR-003**: Shared logic needed by both training workflows and future inference workflows MUST be owned by `aitraf-core` and referenced by dependent surfaces rather than reimplemented separately.
- **FR-004**: The refactor MUST preserve support for the current trick recognition and trick assessment workflows as the first reusable capabilities carried into the new surface split.
- **FR-005**: The system MUST provide a reusable path for producing clip-derived outputs needed by both offline workflows and future serving, including frame-level processing and shared prediction-ready representations.
- **FR-006**: The system MUST support reusable generation of shared model-driven outputs needed across surfaces, including shared feature outputs and shared pose-related outputs where those outputs are part of the existing or planned workflow.
- **FR-007**: The placeholder `aitraf-api` surface MUST remain empty except for package metadata or a short module docstring in this phase.
- **FR-008**: The placeholder `aitraf-api` surface MUST NOT define routes, schemas, operations, adapters, smoke commands, or service runtime behavior in this phase.
- **FR-009**: The refactor MUST distinguish reusable runtime processing from experiment-tracking or registry-managed artifacts so that shared core capabilities are not coupled to training-only artifact ownership.
- **FR-010**: Repository documentation MUST explain how existing maintainers and future API developers discover the correct surface for adding new shared logic, training logic, and serving logic.
- **FR-011**: The system MUST preserve or explicitly document any command, configuration, or package-surface changes required for maintainers to continue preparing data, training models, and evaluating models after the refactor.
- **FR-012**: Missing or incompatible configs, artifacts, model dependencies, or surface contracts introduced by the refactor MUST produce explicit errors with actionable context.

### Architecture And Reuse Requirements *(mandatory)*

- **AR-001**: The feature MUST identify which existing repository surfaces it extends, including `src/aitraf/`, `configs/`, `scripts/`, `Taskfile.yml`, packaging metadata, and repository documentation.
- **AR-002**: The feature MUST reuse the existing repository layout where practical and justify any new parallel structure introduced to represent `aitraf-core`, `aitraf-train`, or `aitraf-api`.
- **AR-003**: Shared clip processing, shared prediction inputs, and other cross-surface runtime capabilities MUST be extracted into reusable modules instead of remaining embedded inside task- or surface-specific flows.
- **AR-004**: Production-relevant behavior for shared processing and future serving contracts MUST live in versioned repository code rather than notebooks or local one-off commands.

### Validation And Reproducibility Requirements *(mandatory)*

- **VR-001**: The feature MUST define validation for repository structure, command continuity, and shared capability reuse, including automated tests where practical and command-level smoke validation for affected workflows.
- **VR-002**: The spec MUST identify the configs, manifests, model references, example inputs, and storage assumptions needed to verify the refactor and prepare future inference work.
- **VR-003**: The refactor MUST describe expected failure behavior for missing surface wiring and missing artifacts, and MUST document that API runtime invocation is unsupported because no API operations are implemented in this phase.
- **VR-004**: If the refactor changes where outputs, caches, or artifacts are produced, the plan MUST identify how maintainers confirm the new ownership boundaries and reproducibility expectations.

### Key Entities *(include if feature involves data)*

- **Repository Surface**: A named in-repo product boundary with a specific responsibility, such as shared runtime capabilities, training orchestration, or future serving entrypoints.
- **Shared Processing Capability**: A reusable processing or prediction-support function that can be invoked from multiple surfaces without duplicating ownership.
- **Inference Operation**: A future externally callable action, initially trick recognition or trick assessment, that will depend on shared core capabilities and explicit runtime contracts in a later phase.
- **Artifact Ownership Boundary**: The rule that determines whether an output belongs to reusable runtime processing or to experiment-tracking and registry-managed workflows.

## Architecture And Data Impact

- **Touched Surfaces**: existing `src/aitraf/` modules re-homed into packages, command entrypoints moved from root `scripts/` into `packages/aitraf-train/scripts/`, configuration surfaces moved from root `configs/` into `packages/aitraf-train/configs/`, task runners in `Taskfile.yml`, package metadata, and repository documentation for developer workflows.
- **Shared Helpers To Add Or Extend**: Shared clip-to-frame processing, reusable prediction-input generation, shared feature and pose output generation, and explicit contracts between shared runtime capabilities and surface-specific orchestration.
- **Data Or Artifact Impact**: Clarify which reusable outputs remain available through repo-managed storage or runtime processing, and which outputs remain owned by experiment-tracking and registry-managed workflows.
- **Reproducibility Inputs**: Existing manifests, task and model configs, documented commands, required model assets, representative clip inputs, and any surface-specific assumptions needed to verify that the refactor preserves expected behavior.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Maintainers can identify the ownership boundary for shared logic, training orchestration, and API entrypoints within 10 minutes using only repository documentation and code layout.
- **SC-002**: 100% of currently supported trick recognition and trick assessment workflows remain runnable after the refactor using documented commands or documented replacements.
- **SC-003**: Shared capabilities required by training support flows are defined once in `aitraf-core`, and future API work has a documented package boundary for composing those capabilities later.
- **SC-004**: The placeholder API surface is present as an empty reserved package and exposes no runtime operations in this phase.

## Assumptions

- The current trick classification workflow is the starting point for future trick recognition behavior.
- The current ordinal score workflow is the starting point for future trick assessment behavior.
- Full production inference is out of scope for this phase; only the repository split, shared-core boundaries, and placeholder API surface are in scope.
- Existing maintainers still need documented prepare, train, and evaluation workflows after the refactor, even if command or package ownership changes.
- Shared runtime processing should remain separate from experiment-tracking or registry-managed outputs unless a future plan explicitly expands that ownership boundary.
