# Feature Specification: Demo Clip Download

**Feature Branch**: `[005-demo-clip-download]`

**Created**: 2026-07-01

**Status**: Draft

**Input**: User description: "Make a high-level specification for downloading demo clips on API load, without duplicating code from aitraf-train and rather reusing aitraf-core."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Demo API Starts With Required Clips (Priority: P1)

An operator starts the API service for the demo environment and expects the demo
videos shown by the API to be available for inference without manually copying
clip files onto the server.

**Why this priority**: The demo API cannot provide working inference if the
selected demo video files are absent from runtime storage.

**Independent Test**: Start the API in an environment with valid object storage
credentials, manifests, and empty runtime clip storage. The demo-selected clips
become available before protected demo inference is considered ready.

**Acceptance Scenarios**:

1. **Given** valid manifests, object storage credentials, and empty clip storage, **When** the API loads with demo clip download enabled, **Then** only the currently selected demo clips are downloaded into runtime clip storage.
2. **Given** selected demo clips already exist in runtime clip storage, **When** the API loads with demo clip download enabled, **Then** existing clips are reused and not redundantly downloaded.
3. **Given** a selected demo clip cannot be fetched, **When** the API loads with demo clip download enabled, **Then** startup fails explicitly and identifies the unavailable clip source.

---

### User Story 2 - Shared Download Behavior Across API And Training (Priority: P2)

A developer maintaining data preparation and serving behavior wants clip download
rules to live in one reusable place so API demo downloads and training data
downloads do not drift.

**Why this priority**: Duplicate download behavior would create inconsistent
failure handling, path handling, and object storage assumptions across serving
and training surfaces.

**Independent Test**: Verify that both the existing training clip download flow
and the new API demo clip download flow use the same shared clip download
capability while preserving their own orchestration responsibilities.

**Acceptance Scenarios**:

1. **Given** the training data pipeline downloads clips from labeled data, **When** the shared download behavior is introduced, **Then** the pipeline continues to download the same clip set to the same storage location.
2. **Given** the API demo flow selects demo clips from API manifests, **When** it requests downloads, **Then** it uses the same shared download behavior as the training flow.

---

### User Story 3 - Restore Image Build Independence From Local Clips (Priority: P3)

A maintainer publishes or pulls the API image and does not expect the image build
to depend on locally available video clips.

**Why this priority**: Published images should remain small and reproducible,
while runtime environments can hydrate the demo clip subset when needed.

**Independent Test**: Remove the current local-clip build-context dependency,
build the API image without local clip files, and verify that image build
succeeds; then validate clip availability through the runtime download flow
instead.

**Acceptance Scenarios**:

1. **Given** the current API Docker build path expects local clips, **When** this feature is implemented, **Then** that build-time clip dependency is removed.
2. **Given** no local clip directory is available during image build, **When** the API image is built, **Then** the build does not attempt to bundle or download clips.
3. **Given** the API image is deployed to a server without clips, **When** runtime download is enabled and credentials are valid, **Then** the server obtains the selected demo clips after startup.

---

### Edge Cases

- Required manifests are missing, empty, unreadable, or do not contain usable
  demo video identifiers.
- A selected demo manifest row lacks an object storage source for the clip.
- Object storage credentials, permissions, or network access are missing.
- Runtime clip storage is missing, read-only, or cannot create destination
  directories.
- Some selected clips already exist locally while others are absent.
- A clip source points to a malformed or unsupported storage location.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST identify the demo clip set from the same demo video selection rules used by the API's demo-video listing.
- **FR-002**: The system MUST make selected demo clips available in the runtime clip storage location before demo inference is considered ready when runtime demo clip download is enabled.
- **FR-003**: The system MUST avoid downloading clips that already exist in the runtime clip storage location unless an explicit refresh behavior is requested.
- **FR-004**: The system MUST fail explicitly when a selected demo clip cannot be resolved, downloaded, or written to runtime storage.
- **FR-005**: The API image build MUST NOT require local clips and MUST NOT download clips during build.
- **FR-006**: The existing training clip download flow MUST continue to support bulk clip downloads for offline data preparation.
- **FR-007**: API demo clip downloads MUST be explicitly enabled by deployment configuration and MUST NOT run as hidden default behavior.

### Architecture And Reuse Requirements *(mandatory)*

- **AR-001**: The feature MUST identify which existing package and feature
  surfaces it extends (`packages/aitraf-api`, `packages/aitraf-core`,
  `packages/aitraf-train`, and their route/service/task/workflow modules).
- **AR-002**: The feature MUST avoid introducing parallel architecture unless the
  spec explicitly justifies why the existing structure cannot be extended.
- **AR-003**: Shared logic MUST be extracted into reusable functions/modules rather
  than duplicated across feature surfaces or task/model pipelines.
- **AR-004**: Production behavior MUST live in versioned repository code, not only
  in notebooks or local ad hoc commands.
- **AR-005**: Business logic MUST prefer functional programming practices where
  practical: pure helpers, explicit inputs and outputs, and localized mutable
  state at framework boundaries.

### Validation And Reproducibility Requirements *(mandatory)*

- **VR-001**: The spec MUST define how the change will be validated, including
  automated tests where practical and command-level smoke validation for pipeline
  behavior.
- **VR-002**: The spec MUST state what configs, manifests, seeds, artifacts, or
  tracking outputs are required to rerun or verify the change.
- **VR-003**: The spec MUST describe expected failure behavior for invalid or
  missing inputs instead of relying on silent fallback paths.
- **VR-004**: If evaluation behavior changes, the spec MUST state which metrics,
  reports, or tracked artifacts are expected to prove correctness.

### Key Entities *(include if feature involves data)*

- **Demo Clip Selection**: The ordered set of demo video identifiers currently
  exposed by the API demo-video listing.
- **Clip Source**: The object storage location associated with a selected demo
  video in the manifest data.
- **Runtime Clip Storage**: The writable storage location where the API expects
  video clips for inference.
- **Shared Clip Download Request**: A reusable download instruction containing a
  clip identifier, source location, destination semantics, and refresh behavior.

## Architecture And Data Impact

- **Touched Surfaces**: `packages/aitraf-api` for runtime demo clip download
  orchestration and demo clip selection; `packages/aitraf-core` for shared clip
  download behavior; `packages/aitraf-train` for reusing the shared download
  behavior from the existing offline data pipeline.
- **Shared Helpers To Add Or Extend**: A reusable clip download capability owned
  by the shared runtime package, plus API-owned demo selection reuse for deciding
  which clips to request.
- **Data Or Artifact Impact**: Uses existing API manifests and existing clip
  source fields. Writes selected demo clips into runtime clip storage. Does not
  alter labels, vocabularies, model artifacts, metrics, or evaluation outputs.
- **Reproducibility Inputs**: API manifests, object storage credentials, runtime
  clip storage path, selected demo video rules, and the existing data pipeline's
  source clip metadata.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In a clean runtime storage location with valid credentials, 100% of currently selected demo clips are available before demo inference is considered ready.
- **SC-002**: Re-running startup with the same selected clips already present completes without re-downloading those clips.
- **SC-003**: The existing offline clip download flow continues to download the same clip set as before for a representative labels input.
- **SC-004**: The API image can be built in an environment with no local clip files.
- **SC-005**: When one selected demo clip source is invalid or inaccessible, the failure identifies the affected clip and does not silently remove it from the demo set.

## Assumptions

- Demo clips are the same videos returned by the API demo-video selection rules.
- Selected demo manifest rows include enough source metadata to locate each clip
  in object storage.
- Runtime deployments that enable demo clip download provide the required object
  storage credentials and writable clip storage.
- Runtime demo clip download is optional and explicitly configured; deployments
  that mount prehydrated storage may keep it disabled.
- The existing training data pipeline remains the owner of offline data
  orchestration; only reusable clip download behavior is promoted to the shared
  runtime package.
- Invalid or partial runtime clip state is surfaced as an explicit error rather
  than repaired or hidden.
