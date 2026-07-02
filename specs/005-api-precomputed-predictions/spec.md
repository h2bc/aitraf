# Feature Specification: AITRAF API Precomputed Predictions

**Feature Branch**: `[005-api-precomputed-predictions]`

**Created**: 2026-07-02

**Status**: Draft

**Input**: User description: "simplify aitraf-api demo serving to get test
predictions from MLOps and serve them directly instead of running inference, due
to limited server compute, offline non-generalized demo models, and demo-only
selected test videos with no uploaded video inference."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Serve Demo Results Without Inference (Priority: P1)

As a demo operator, I need the demo API to return already-computed demo
predictions for videos with matching prediction rows so the demo can run
reliably on limited server resources without depending on live model execution.

**Why this priority**: The demo only supports a preselected subset of test-set
videos. Running the offline demo models again during serving adds cost and
failure risk without adding product value.

**Independent Test**: Start the demo API with a valid precomputed prediction
source, request the demo-predictions list, and verify every returned item
includes the expected prediction data without requiring live prediction work.

**Acceptance Scenarios**:

1. **Given** valid precomputed prediction artifacts are available, **When** the
   demo API starts, **Then** the artifacts are downloaded and read before demo
   traffic is accepted.
2. **Given** an authorized user opens the demo, **When** the user requests demo
   predictions, **Then** the API returns prediction records for videos with
   matching precomputed predictions.
3. **Given** a prediction artifact is unavailable or unreadable, **When** the
   demo API starts, **Then** startup or readiness fails explicitly instead of
   running live inference or returning partial data.

---

### User Story 2 - Keep Predictions Tied To MLOps Results (Priority: P2)

As an ML maintainer, I need served demo predictions to come from tracked test
prediction artifacts so the demo uses the saved MLOps outputs instead of new
serving-time computation.

**Why this priority**: The models are offline demo models rather than general
online inference models. The source of truth for demo predictions should be the
tracked evaluation output, not a new serving-time computation.

**Independent Test**: Provide prediction artifacts from the tracked evaluation
system and verify the API downloads them, reads prediction rows, and builds a
demo predictions response from videos with matching prediction rows.

**Acceptance Scenarios**:

1. **Given** tracked full test-prediction artifacts exist for the demo model
   outputs, **When** their run IDs are supplied to the API, **Then** the API
   downloads the artifacts from fixed code-level artifact paths and reads their
   prediction rows.
2. **Given** prediction artifacts follow the eval export contract, **When** the
   API starts or serves demo predictions, **Then** the API uses those rows
   directly without running extra row-schema validation.
3. **Given** prediction artifacts are generated outputs, **When** repository
   changes are reviewed, **Then** generated prediction files are not committed
   as source files.

---

### User Story 3 - Remove Unsupported Online Inference Behavior (Priority: P3)

As a product maintainer, I need the API surface to reflect that the demo does
not accept uploaded or arbitrary videos so users and operators do not rely on an
unsupported live-inference workflow.

**Why this priority**: Keeping unused online-inference behavior makes the demo
harder to operate and suggests a generalization capability the current offline
models do not provide.

**Independent Test**: Verify the simplified API exposes demo predictions only
through `/demo-predictions` and does not keep arbitrary per-video inference
routes.

**Acceptance Scenarios**:

1. **Given** a video is not part of the matched demo predictions response,
   **When** demo predictions are requested, **Then** that video is not returned.
2. **Given** old serving configuration points at live model inputs only, **When**
   the simplified demo API is started, **Then** that configuration is rejected as
   incomplete for the new demo-serving mode.
3. **Given** implementation is complete, **When** documentation and validation
   are reviewed, **Then** they describe precomputed demo prediction serving
   rather than live uploaded-video inference.

### Edge Cases

- Required prediction source is missing, unavailable, unreadable, or empty.
- Prediction source contains malformed records.
- Prediction source rows lack fields needed for filtering or display.
- Classification and AQA prediction sources contain non-overlapping video IDs.
- Old per-video inference routes are requested after they have been removed.
- Existing live-inference configuration is supplied without precomputed
  prediction configuration.
- External MLOps access is unavailable during startup or release preparation.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The demo API MUST serve predictions from precomputed test
  prediction results rather than performing live inference.
- **FR-002**: The system MUST load prediction results only from an explicitly
  configured prediction source.
- **FR-003**: The system MUST check that configured prediction artifacts can be
  downloaded, parsed, and filtered by `video_id` before accepting demo traffic.
- **FR-004**: The API MUST build the served demo predictions response from video
  metadata embedded in the full prediction artifacts and matching prediction
  rows.
- **FR-005**: The prediction source MUST be configured explicitly with MLflow
  run IDs. Prediction artifact paths MUST be fixed constants in API code, not
  environment variables.
- **FR-006**: The API MUST remove arbitrary per-video live prediction routes
  from the demo serving surface.
- **FR-007**: The API MUST fail explicitly when prediction results are missing,
  malformed, incomplete, unreadable, or unavailable.
- **FR-008**: The API MUST NOT silently fall back to live inference when
  precomputed prediction results cannot be loaded.
- **FR-009**: Generated prediction result files MUST remain outside source
  control.
- **FR-010**: Documentation MUST describe the demo-serving limitation: only demo
  prediction records with matching precomputed predictions are returned.
- **FR-011**: The root `.env.example` MUST be refactored for the simplified API
  by removing API live model URI variables and adding prediction run ID
  variables without prediction artifact path variables. Shared train/runtime
  path variables still used by code MUST remain available for offline workflows.
- **FR-012**: All `aitraf-train` evaluation scripts MUST support logging full
  test prediction artifacts, not only metrics, plots, params, or misses.
- **FR-013**: The final API validation MUST use MLflow run IDs that contain full
  `test_predictions.json` artifacts, not the old inspected eval runs and not
  training run IDs. Producing those run IDs is an external/manual step.

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
- **AR-006**: The feature MUST update callers, tests, docs, and validation
  commands to the new behavior directly and remove obsolete paths, aliases,
  shims, deprecated parameters, compatibility layers, and dead code.
- **AR-007**: Prediction artifact loading MUST accept one required
  `test_predictions.json` record schema and fail on mismatches instead of
  accepting alternate shapes, broad unions, stringified structured data, or
  fallback conversion branches.

### Validation And Reproducibility Requirements *(mandatory)*

- **VR-001**: The feature MUST be validated with automated checks for valid
  artifact loading, unreadable artifact failure, and matched demo-predictions
  responses.
- **VR-002**: The feature MUST include command-level smoke validation proving the
  demo can run from a valid precomputed prediction source.
- **VR-003**: The feature MUST document which prediction run IDs, fixed artifact
  paths, and runtime configuration are required to verify the served results.
- **VR-004**: The feature MUST describe expected failure behavior for missing
  sources, unavailable MLOps access, and invalid records.

### Key Entities *(include if feature involves data)*

- **Precomputed Prediction Source**: The configured source of tracked test
  predictions used by the demo API. Key attributes include task name, MLflow run
  ID, fixed code-level artifact path, and availability status.
- **Demo Prediction**: One returned demo prediction record and its associated task
  predictions. Key attributes include video identifier, display metadata,
  ground-truth labels when available, prediction labels, confidence values, and
  task names.
- **Prediction Artifact Row**: One row loaded from a downloaded prediction
  artifact. Key attributes include video identifier, prediction label, optional
  confidence, video display metadata, and optional artifact-provided metadata.
- **Matched Demo Predictions Response**: The final in-memory response built at
  startup from downloaded classification and AQA prediction rows with matching
  video identifiers.

## Architecture And Data Impact

- **Touched Surfaces**: API demo-predictions serving behavior under
  `packages/aitraf-api`; required train/evaluation artifact export behavior
  under `packages/aitraf-train` because existing MLOps outputs do not provide
  full prediction artifacts; API documentation and validation commands.
- **Shared Helpers To Add Or Extend**: Artifact download/read helpers, simple
  row-field checks, startup matching/filtering, and response mapping should be
  reusable within the API surface without duplicating task-specific parsing.
- **Legacy Surfaces Removed**: Demo-serving dependencies on live model loading,
  serving-time video decoding, serving-time feature extraction, arbitrary-video
  prediction, and live-inference runtime configuration.
- **Data Or Artifact Impact**: The API depends on tracked precomputed prediction
  artifacts from MLOps or release materialization. Generated prediction results
  are not source files. Existing model training is not changed by this feature.
- **Reproducibility Inputs**: Configured prediction run IDs, fixed artifact
  paths in API code, runtime demo configuration, and validation commands.
- **Validation Prediction Run IDs**: Classification
  `2b2208e417e34e2198bb108e4f683cf9` and AQA
  `da6a8082c5e646448c7a79cd124b8e09`. Both contain full
  `test_predictions.json` artifacts with the required API metadata and
  prediction fields.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of demo prediction records returned by the API include complete
  precomputed predictions for the supported demo tasks.
- **SC-002**: Old arbitrary per-video live inference routes are no longer
  available from the demo API.
- **SC-003**: 100% of covered missing, malformed, incomplete, unreadable, or
  unavailable prediction sources fail explicitly.
- **SC-004**: Demo startup no longer requires live model execution resources for
  the supported demo prediction path.
- **SC-005**: Generated prediction outputs remain out of source control while
  the API is configured from explicit tracked MLflow run IDs and fixed artifact
  paths in code.
- **SC-006**: The completed implementation is validated by building the Docker
  image and, once external run IDs are supplied, running that container so it
  downloads both MLflow `test_predictions.json` artifacts at startup, then smoke
  testing `/health` plus authenticated `GET /demo-predictions`.
- **SC-007**: Full prediction artifacts are produced by evaluation runs for the
  demo models. The API implementation consumes the resulting run IDs and does
  not own running those evaluations.

## Assumptions

- The demo is intentionally limited to videos that exist in both downloaded
  prediction artifacts.
- The demo does not support arbitrary uploaded-video prediction.
- The current demo models are offline and not intended to serve as generalized
  online inference models.
- Current inspected eval runs do not contain full prediction artifacts. Suitable
  test prediction results must be supplied from evaluation runs after export
  logging is added.
- Invalid or unavailable prediction state is surfaced as an explicit error
  rather than repaired, guessed, or replaced with live inference.
