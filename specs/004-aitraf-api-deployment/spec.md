# Feature Specification: AITRAF API Deployment

**Feature Branch**: `[004-aitraf-api-deployment]`

**Created**: 2026-06-25

**Status**: Draft

**Input**: User description: "we need to set up the deployment for aitraf-api. basically we need a prod aitraf-api docker image and to exten the worfklow to build it on push to master. Thats the idea. P1 i think its docker file. P2 is github workflow. thats it."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Create API Dockerfile (Priority: P1)

As an API operator, I need a production Dockerfile for `aitraf-api` so the serving package can be built into a repeatable deployment image from repository sources.

**Why this priority**: The Dockerfile is the base deployment artifact. Without it, the API cannot be built consistently for production.

**Independent Test**: Can be tested by building the API image from the Dockerfile, starting the service with required runtime settings, and confirming the service exposes its expected availability signal.

**Acceptance Scenarios**:

1. **Given** the repository contains the API package and its repository-owned dependencies, **When** the API Dockerfile is built, **Then** the resulting image contains the API service and can start the API entrypoint.
2. **Given** all required API runtime settings and credentials are supplied outside the image, **When** the image starts, **Then** the service becomes available for health validation.
3. **Given** required runtime settings, model references, or storage paths are missing, **When** the image starts or validates readiness, **Then** the failure is explicit and does not silently substitute defaults.

---

### User Story 2 - Publish API Image From GitHub Workflow (Priority: P2)

As a maintainer, I need a GitHub workflow that builds and publishes the `aitraf-api` image on pushes to `master` so production candidates are tied to reviewed source revisions.

**Why this priority**: Once the Dockerfile exists, workflow automation removes manual publishing drift and makes new master revisions deployable.

**Independent Test**: Can be tested by triggering the GitHub workflow for the `master` branch and verifying a new API image is produced, published, and traceable to the source revision.

**Acceptance Scenarios**:

1. **Given** a change is pushed to `master`, **When** the GitHub workflow runs, **Then** it builds and publishes an `aitraf-api` image.
2. **Given** a maintainer needs to reproduce publishing without a source change, **When** they manually trigger the workflow, **Then** it builds and publishes the same API image type.
3. **Given** the build, authentication, or publish step fails, **When** the workflow completes, **Then** the workflow reports failure and no successful deployment candidate is advertised for that revision.
4. **Given** a published API image exists, **When** an operator inspects its tags or metadata, **Then** they can determine the source revision it was built from.

### Edge Cases

- The Dockerfile omits the API package, entrypoint, or required repository-owned runtime dependency.
- Required secrets, model references, data paths, storage paths, or tracking credentials are not available at runtime.
- The publish workflow lacks permission to write the API image.
- The source change is pushed to a branch other than `master`.
- The published image tag already exists or cannot be associated with the intended source revision.
- Build caching or previous artifacts could mask a broken current API image build.
- The existing train image workflow exists, but the API workflow drifts in naming, triggers, permissions, or traceability conventions.
- The API image builds successfully but cannot pass a health or smoke validation with required runtime inputs.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a production Dockerfile for `aitraf-api`.
- **FR-002**: The Dockerfile MUST build an API image that includes the API service code and all repository-owned runtime code required by the serving package.
- **FR-003**: The API image MUST NOT include committed secrets or environment-specific credentials.
- **FR-004**: The API image MUST require explicit runtime settings for API auth, data access, storage access, model references, and model tracking access before protected serving behavior is considered ready.
- **FR-005**: System MUST provide a GitHub workflow that builds and publishes the `aitraf-api` image on every push to `master`.
- **FR-006**: The GitHub workflow MUST allow maintainers to manually run the same image publishing path.
- **FR-007**: Published API images MUST include a stable latest reference and a source-revision-specific reference.
- **FR-008**: The GitHub workflow MUST fail visibly when build, authentication, permission, or publish requirements are not met.
- **FR-009**: The GitHub workflow MUST avoid advertising a successful API deployment candidate when the image build or publish step fails.
- **FR-010**: Documentation or validation instructions MUST state the runtime settings and smoke validation path required to verify the API image.
- **FR-011**: The API image publishing process MUST follow existing repository image publishing conventions where those conventions apply.
- **FR-012**: The feature MUST include validation coverage or command-level checks proving the API Dockerfile builds and the GitHub workflow is ready for the repository.

### Architecture And Reuse Requirements *(mandatory)*

- **AR-001**: The feature MUST extend `packages/aitraf-api` for the API Dockerfile and repository workflow surfaces for image publishing.
- **AR-002**: The feature MUST avoid introducing a parallel deployment architecture when existing repository image publishing conventions can be extended.
- **AR-003**: Shared build or publishing patterns already used by repository image workflows MUST be reused or mirrored rather than duplicated with incompatible behavior.
- **AR-004**: Production behavior MUST live in versioned repository code, not only in notebooks or local ad hoc commands.
- **AR-005**: Deployment configuration MUST keep secrets outside committed files and require explicit runtime inputs.

### Validation And Reproducibility Requirements *(mandatory)*

- **VR-001**: The feature MUST be validated with command-level checks for building the API Dockerfile and verifying the GitHub workflow definition.
- **VR-002**: The feature MUST document the runtime environment settings, model references, data paths, storage paths, and credentials required to run or verify the API service from the image.
- **VR-003**: The feature MUST describe expected failure behavior for missing runtime inputs, missing API entrypoint, Dockerfile build failures, workflow permission failures, and publish failures instead of relying on silent fallback paths.
- **VR-004**: If API behavior changes during deployment packaging, existing API tests or smoke validation MUST confirm health, auth, demo-video listing, and inference behavior remain available.

### Key Entities *(include if feature involves data)*

- **API Dockerfile**: The versioned build definition for the `aitraf-api` deployment image. Key attributes include package identity, runtime entrypoint, included repository-owned dependencies, and required runtime inputs.
- **API Deployment Image**: The deployable artifact for the `aitraf-api` serving package. Key attributes include package identity, source revision, runtime entrypoint, required runtime inputs, and publish references.
- **GitHub Image Publishing Workflow**: Repository automation that produces and publishes the API deployment image. Key attributes include trigger branch, manual trigger support, publish destination, required permissions, tags, and failure status.
- **Runtime Configuration**: Environment-provided values required by the API service. Key attributes include auth token, data path, storage path, model references, tracking location, and external credentials.

## Architecture And Data Impact

- **Touched Surfaces**: Add production Dockerfile support under `packages/aitraf-api`, extend repository workflow configuration for API image publishing, and update API documentation or validation notes as needed.
- **Shared Helpers To Add Or Extend**: Reuse the existing repository image publishing pattern for trigger behavior, artifact naming, permissions, metadata, and publish traceability. No new shared runtime helper is expected.
- **Data Or Artifact Impact**: Creates a published API deployment image artifact. Does not alter training data, manifests, model artifacts, labels, metrics, or evaluation outputs.
- **Reproducibility Inputs**: Repository source revision, API package metadata, API Dockerfile, GitHub image publishing workflow, required runtime environment settings, model references, data and storage paths, model tracking location, credentials supplied outside committed files, and documented smoke validation steps.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A developer can build the `aitraf-api` image from the production Dockerfile using repository sources.
- **SC-002**: A developer can start the built API image and reach the API health validation path using documented runtime inputs.
- **SC-003**: 100% of pushes to `master` that complete the GitHub workflow successfully produce an `aitraf-api` image.
- **SC-004**: 100% of published API images include both a stable latest reference and a source-revision-specific reference.
- **SC-005**: Missing required runtime settings are surfaced as explicit startup or readiness failures in 100% of covered validation cases.
- **SC-006**: Existing validated API behavior remains available after packaging, including health, auth enforcement, demo-video listing, and supported inference smoke paths.

## Assumptions

- The deployment target consumes a production API image built from a Dockerfile and published by GitHub workflow automation.
- The existing train image publishing workflow is the baseline convention for registry, trigger, tag, permission, and metadata behavior unless planning identifies a documented reason to diverge.
- The publishing branch is `master`, matching the user's requested workflow trigger.
- Runtime secrets and deployment credentials are supplied by the deployment environment or repository secret store, not committed files.
- This feature does not provision hosting infrastructure, external networking, data volumes, model registry contents, or deployment rollout policy.
- Missing configuration, unavailable artifacts, invalid permissions, and failed publish attempts are explicit failures rather than silently repaired or skipped.
