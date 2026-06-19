# Feature Specification: API Inference Surface

**Feature Branch**: `[002-api-inference]`

**Created**: 2026-06-19

**Status**: Draft

**Input**: User description: "Implement an API with endpoints for API health, demo videos, trick classification inference, and trick AQA inference. Demo videos should return matching rows from the current trick classification and trick AQA test set manifests for later display in a Next.js demo UI. Inference endpoints should accept a video id or existing test set index and run predictions. The API must use token-based auth so only approved apps can call it, and include tests for basic functionality."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Verify API Availability (Priority: P1)

As a demo frontend operator, I need to confirm the inference service is reachable before users attempt predictions, so the demo UI can fail clearly when the service is unavailable.

**Why this priority**: Availability is the first dependency for every downstream demo action.

**Independent Test**: Can be tested by calling the health check and verifying it returns a successful service status without requiring a prediction request.

**Acceptance Scenarios**:

1. **Given** the API service is running, **When** an authorized app requests the health check, **Then** the app receives a successful response that indicates the service is available.
2. **Given** the API service is not available, **When** the frontend attempts to check health, **Then** the frontend can detect the failure from the request outcome.

---

### User Story 2 - List Demo Video Choices (Priority: P1)

As a demo frontend, I need to fetch the currently supported demo video rows for trick classification and trick AQA so a user can choose an existing test-set video before running inference.

**Why this priority**: Prediction requests depend on selecting a valid test-set example; listing demo choices provides a bounded and reproducible input set.

**Independent Test**: Can be tested by requesting demo videos and confirming the response contains rows from both supported current test manifests with stable identifiers that can be reused for inference.

**Acceptance Scenarios**:

1. **Given** current test manifests exist for trick classification and trick AQA, **When** an authorized app requests demo videos, **Then** the app receives matching demo rows for both tasks.
2. **Given** the app selects one returned demo row, **When** it sends the row identifier or index to the matching inference capability, **Then** the identifier is accepted as a valid test-set selection.
3. **Given** a required manifest is missing or unreadable, **When** demo videos are requested, **Then** the API returns an explicit failure rather than silently returning partial or fabricated rows.

---

### User Story 3 - Run Trick Classification Prediction (Priority: P1)

As a demo frontend, I need to request a trick classification prediction for a selected test-set video so the demo can show the predicted trick class.

**Why this priority**: Trick classification is one of the two required demo prediction outcomes.

**Independent Test**: Can be tested by choosing a valid classification test-set index and verifying the response contains a classification result tied to the selected row.

**Acceptance Scenarios**:

1. **Given** an authorized app submits a valid classification test-set index, **When** prediction is requested, **Then** the response contains the selected input identity and a classification prediction.
2. **Given** an authorized app submits an index outside the current classification test set, **When** prediction is requested, **Then** the request fails with a clear invalid-selection error.
3. **Given** required model artifacts or input assets are unavailable, **When** prediction is requested, **Then** the request fails explicitly and does not return a guessed prediction.

---

### User Story 4 - Run Trick AQA Prediction (Priority: P1)

As a demo frontend, I need to request a trick AQA prediction for a selected test-set video so the demo can show the predicted quality score or assessment output.

**Why this priority**: Trick AQA is the second required demo prediction outcome and must use the same reproducible test-set-selection model.

**Independent Test**: Can be tested by choosing a valid AQA test-set index and verifying the response contains an AQA result tied to the selected row.

**Acceptance Scenarios**:

1. **Given** an authorized app submits a valid AQA test-set index, **When** prediction is requested, **Then** the response contains the selected input identity and an AQA prediction.
2. **Given** an authorized app submits an index outside the current AQA test set, **When** prediction is requested, **Then** the request fails with a clear invalid-selection error.
3. **Given** required model artifacts or input assets are unavailable, **When** prediction is requested, **Then** the request fails explicitly and does not return a guessed prediction.

---

### User Story 5 - Restrict API Access (Priority: P1)

As the API owner, I need only approved frontend apps with the configured token to call protected API capabilities so demo inference is not exposed publicly.

**Why this priority**: The API is intended for a controlled frontend integration, not anonymous use.

**Independent Test**: Can be tested by calling each protected capability without a token, with an invalid token, and with the approved token.

**Acceptance Scenarios**:

1. **Given** a request has no token, **When** it calls a protected capability, **Then** the API rejects the request.
2. **Given** a request has an invalid token, **When** it calls a protected capability, **Then** the API rejects the request.
3. **Given** a request has the approved token, **When** it calls a protected capability with valid inputs, **Then** the API processes the request.

### Edge Cases

- Required test manifests are missing, empty, unreadable, or missing expected selection columns.
- A caller provides a negative, non-integer, or out-of-range test-set index.
- A caller provides a video identifier that does not match the current manifest row for the requested task.
- Model artifacts, derived features, labels, or input video assets required for prediction are missing.
- Token configuration is absent when protected capabilities are used.
- A supported task is requested with the other task's index or identifier.
- Manifest schema or label vocabulary changes from the model artifact expectations.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a health capability that returns whether the API service is available.
- **FR-002**: System MUST provide demo video listings from the current trick classification and trick AQA test manifests.
- **FR-003**: Demo video listings MUST include enough stable row identity for the frontend to request inference for the selected current test-set row.
- **FR-004**: System MUST provide trick classification inference for a selected current test-set row.
- **FR-005**: System MUST provide trick AQA inference for a selected current test-set row.
- **FR-006**: Inference selection MUST support the existing test-set index as the canonical request input.
- **FR-007**: Inference responses MUST include the selected input identity and prediction output so the frontend can associate results with the selected video.
- **FR-008**: Protected API capabilities MUST reject requests that do not provide the approved token.
- **FR-009**: Protected API capabilities MUST reject requests that provide an invalid token.
- **FR-010**: The API MUST return explicit failures for missing manifests, invalid selections, missing artifacts, and unsupported task requests.
- **FR-011**: The API MUST include automated coverage for health checks, token enforcement, demo video listing, valid inference requests, and invalid selection handling.
- **FR-012**: The API MUST keep inference inputs bounded to current test-set rows and MUST NOT accept arbitrary uploaded videos in this feature.

### Architecture And Reuse Requirements *(mandatory)*

- **AR-001**: The feature MUST identify which existing repository surfaces it extends (`configs/`, `scripts/`, `src/aitraf/tasks/`, shared processing/utilities).
- **AR-002**: The feature MUST avoid introducing parallel architecture unless the spec explicitly justifies why the existing structure cannot be extended.
- **AR-003**: Shared logic MUST be extracted into reusable functions/modules rather than duplicated across task/model pipelines.
- **AR-004**: Production behavior MUST live in versioned repository code, not only in notebooks or local ad hoc commands.

### Validation And Reproducibility Requirements *(mandatory)*

- **VR-001**: The spec MUST define how the change will be validated, including automated tests where practical and command-level smoke validation for pipeline behavior.
- **VR-002**: The spec MUST state what configs, manifests, seeds, artifacts, or tracking outputs are required to rerun or verify the change.
- **VR-003**: The spec MUST describe expected failure behavior for invalid or missing inputs instead of relying on silent fallback paths.
- **VR-004**: If evaluation behavior changes, the spec MUST state which metrics, reports, or tracked artifacts are expected to prove correctness.

### Key Entities *(include if feature involves data)*

- **Demo Video Row**: A row from a current task test manifest that can be displayed in the demo UI and selected for inference. Key attributes include task, test-set index, video identity, source row metadata, and any display-safe label or score fields already present in the manifest.
- **Inference Request**: A protected request from an approved app that selects a current test-set row for a supported task. Key attributes include task, test-set index, optional video identity for consistency checking, and request authorization token.
- **Inference Result**: A prediction response for one selected test-set row. Key attributes include task, selected test-set index, selected video identity, prediction output, and any model/result metadata needed to interpret the response.
- **API Token**: A configured shared secret used to authorize approved app requests. It is not stored in committed repository files.

## Architecture And Data Impact

- **Touched Surfaces**: Extend the API package reserved by the surface split; reuse current task manifests and existing model/runtime processing surfaces needed to produce trick classification and trick AQA predictions; add tests for the API package.
- **Shared Helpers To Add Or Extend**: Add reusable manifest selection, response shaping, token validation, and task inference orchestration helpers rather than duplicating task-specific request handling.
- **Data Or Artifact Impact**: Reads current trick classification and trick AQA test manifests plus any existing prediction artifacts, labels, vocabularies, features, or model outputs required for inference; does not create new training data or alter manifests.
- **Reproducibility Inputs**: Current test manifests for both tasks, existing model artifacts and vocabularies for each prediction task, configured API token, and documented command/test entrypoints for verifying the API behavior.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An approved frontend app can complete health, demo listing, and each supported inference flow using current test-set rows with no manual backend intervention.
- **SC-002**: 100% of protected capabilities reject missing or invalid tokens during automated validation.
- **SC-003**: 100% of invalid test-set index cases covered by automated tests return explicit invalid-selection failures.
- **SC-004**: Demo video listing returns rows for both supported tasks whenever both current test manifests are present.
- **SC-005**: Basic API functionality is covered by automated tests that can be run by another developer from the repository.

## Assumptions

- The Next.js frontend is the only intended consumer for this feature, but token validation should be generic enough for any approved app using the configured token.
- The current test-set index is the canonical inference input; video identity may be included to help the frontend verify selections.
- Trick AQA maps to the repository's existing score/quality-assessment task surface and current test manifest.
- Health checking may be available without authentication if needed for deployment monitoring, while demo listing and inference are protected.
- Missing manifests, missing artifacts, schema mismatches, and invalid selections are surfaced as explicit errors rather than repaired or silently skipped.
- This feature does not support arbitrary uploaded videos, retraining, manifest generation, model registry mutation, or evaluation metric changes.
