# Feature Specification: Temporal Fusion Trick AQA

**Feature Branch**: `[003-temporal-fusion-aqa]`

**Created**: 2026-06-20

**Status**: Draft

**Input**: User description: "We need to add a temporal fusion model for trick aqa in the API. The goal of the user is to predict trick aqa with a temporal fusion model"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Predict Trick AQA With Temporal Fusion (Priority: P1)

As a demo frontend user, I need the trick AQA prediction flow to use the temporal fusion model so the displayed assessment comes from the intended model family.

**Why this priority**: This is the primary user outcome; without it, the trick AQA capability does not satisfy the requested model behavior.

**Independent Test**: Can be tested by selecting a valid trick AQA demo video and confirming the response contains an assessment prediction produced by the configured temporal fusion model, tied to the selected video.

**Acceptance Scenarios**:

1. **Given** the temporal fusion model is available for trick AQA and a valid AQA demo video is selected, **When** an approved app requests the prediction, **Then** the app receives a trick AQA result identified as coming from the temporal fusion model.
2. **Given** a valid AQA demo video is selected, **When** prediction completes, **Then** the response includes the selected video identity, predicted AQA output, available ground-truth assessment, and model identity needed for display.
3. **Given** the existing trick classification flow is available, **When** trick AQA temporal fusion prediction is added, **Then** trick classification behavior remains unchanged.

---

### User Story 2 - Reuse Core Processing Across Training And Serving (Priority: P1)

As a developer maintaining trick AQA, I need temporal fusion training and API inference to reuse the same shared runtime processing from `aitraf-core` so frame sampling, feature extraction, and preprocessing behavior stay consistent.

**Why this priority**: Temporal fusion depends on video-derived inputs; duplicating that logic in training and serving would create drift and make predictions difficult to reproduce.

**Independent Test**: Can be tested by validating that both temporal fusion training/evaluation and API prediction use the shared `aitraf-core` frame sampling and VideoMAE feature interfaces rather than task-local copies.

**Acceptance Scenarios**:

1. **Given** temporal fusion training or evaluation prepares video inputs, **When** it samples frames or derives VideoMAE features, **Then** it uses shared `aitraf-core` functionality.
2. **Given** the API prepares a trick AQA temporal fusion prediction, **When** it samples frames or derives VideoMAE features, **Then** it uses the same shared `aitraf-core` functionality as training/evaluation.
3. **Given** a preprocessing behavior changes in `aitraf-core`, **When** training/evaluation and API prediction are validated, **Then** both surfaces observe the same behavior without duplicate updates.

---

### User Story 3 - Cache Reusable Temporal Fusion Inputs (Priority: P2)

As a developer or demo operator, I need model instances, sampled frames, and VideoMAE features to be cached when configured so repeated trick AQA predictions avoid unnecessary repeated work while remaining reproducible.

**Why this priority**: Temporal fusion inference can be expensive for repeated demo videos; explicit caching improves usability without changing prediction semantics.

**Independent Test**: Can be tested by making repeated prediction requests for the same valid trick AQA video with caching enabled and verifying the same result is returned while reusable model, frame, and feature work is not repeated.

**Acceptance Scenarios**:

1. **Given** a temporal fusion model is loaded for trick AQA, **When** additional trick AQA predictions are requested, **Then** the model instance is reused from a controlled cache rather than reloaded for every request.
2. **Given** frame-sampling cache is enabled and the same video is requested again with the same sampling contract, **When** frames are needed, **Then** cached sampled frames are reused.
3. **Given** VideoMAE feature cache is enabled and the same video features are requested again with the same feature contract, **When** temporal fusion needs those features, **Then** cached features are reused.
4. **Given** any cache is disabled or invalid for the requested input contract, **When** prediction is requested, **Then** the system recomputes the required value explicitly and does not use stale cached data.

### Edge Cases

- The temporal fusion model reference is absent, malformed, or points to an unsupported task.
- The model artifact is missing required preprocessing, scoring, decoding, or display metadata.
- A requested video id or test-set row is not part of the current trick AQA manifest.
- The selected video file or derived inputs are missing from storage.
- The model output cannot be decoded into the expected trick AQA assessment representation.
- The configured trick AQA model kind conflicts with the model artifact's declared task or output contract.
- Authenticated callers request trick AQA while the temporal fusion model is not ready for serving.
- Frame sampling cache is disabled, empty, stale, or incompatible with the requested sampling contract.
- VideoMAE feature cache is disabled, empty, stale, or incompatible with the requested feature contract.
- Cached model, frame, or feature entries exist for a different model version, video identity, preprocessing contract, or feature contract.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST use the temporal fusion model as the prediction source for trick AQA requests.
- **FR-002**: System MUST keep trick AQA prediction inputs bounded to the current supported demo/test-set video selections.
- **FR-003**: Trick AQA prediction responses MUST include selected video identity, predicted assessment output, available ground-truth assessment, and compact temporal fusion model identity.
- **FR-004**: System MUST reject trick AQA prediction requests when the selected video is invalid for the current trick AQA manifest.
- **FR-005**: System MUST reject trick AQA prediction requests when the temporal fusion model, required model metadata, input assets, or output decoding information is missing or incompatible.
- **FR-006**: System MUST NOT substitute an ordinal, baseline, classification, dummy, or previously cached prediction when temporal fusion prediction cannot be completed.
- **FR-007**: System MUST preserve existing health, demo-video listing, auth enforcement, and trick classification behaviors while adding temporal fusion trick AQA prediction.
- **FR-008**: System MUST expose only display-safe model identity and assessment interpretation data in prediction responses.
- **FR-009**: System MUST reuse `aitraf-core` frame sampling and VideoMAE feature functionality from both temporal fusion training/evaluation and API prediction.
- **FR-010**: System MUST cache loaded temporal fusion model instances for repeated API predictions within a controlled runtime cache.
- **FR-011**: System MUST support an explicit frame-sampling cache flag that controls whether sampled frames are reused for matching video and sampling contracts.
- **FR-012**: System MUST support VideoMAE feature caching for matching video, model, preprocessing, and feature contracts.
- **FR-013**: System MUST invalidate or bypass cached frames and features when their stored contract does not match the requested prediction contract.
- **FR-014**: System MUST include validation coverage for a successful temporal fusion trick AQA prediction, invalid AQA selection, unavailable temporal fusion model, prevention of silent fallback, shared `aitraf-core` processing reuse, model caching, frame-sampling cache behavior, and VideoMAE feature cache behavior.

### Architecture And Reuse Requirements *(mandatory)*

- **AR-001**: The feature MUST extend `packages/aitraf-api` for serving behavior, `packages/aitraf-core` for reusable runtime processing and cacheable frame/feature helpers, and `packages/aitraf-train` only for temporal fusion training, evaluation, registration, configuration, and artifact metadata work if those surfaces require updates.
- **AR-002**: The feature MUST avoid introducing a parallel trick AQA serving path and MUST extend the existing trick AQA route/service surface.
- **AR-003**: Shared model loading, artifact validation, response shaping, manifest selection, frame sampling, and VideoMAE feature logic MUST be reusable helpers rather than duplicated temporal-fusion-only request handling.
- **AR-004**: Production behavior MUST live in versioned repository code and configuration, not only in notebooks or local ad hoc commands.
- **AR-005**: Business logic MUST prefer explicit inputs and outputs, transformation-oriented helpers, and localized serving state at runtime boundaries.

### Validation And Reproducibility Requirements *(mandatory)*

- **VR-001**: The feature MUST be validated with automated coverage where practical and a documented smoke flow that proves temporal fusion trick AQA prediction can be requested for a valid demo video.
- **VR-002**: The feature MUST state the required model reference, data manifests, video storage inputs, serving metadata, decoding metadata, cache settings, cache contract metadata, and credentials needed to rerun or verify the prediction flow.
- **VR-003**: The feature MUST surface invalid selections, missing configuration, missing artifacts, missing input assets, incompatible model outputs, and incompatible cache entries as explicit errors or explicit recomputation paths rather than silent fallback paths.
- **VR-004**: If temporal fusion training or evaluation artifacts are updated, validation MUST identify the tracked metrics or reports used to confirm that the registered model is suitable for serving.

### Key Entities *(include if feature involves data)*

- **Trick AQA Demo Video**: A current supported trick AQA test-set row that can be selected for prediction. Key attributes include row identity, video identity, source metadata, and available assessment label or score.
- **Temporal Fusion Model Artifact**: The registered prediction asset used for trick AQA. Key attributes include task identity, model identity, required preprocessing metadata, output decoding metadata, and display-safe model kind.
- **Trick AQA Prediction Request**: An approved app request selecting one current trick AQA demo video for assessment prediction.
- **Trick AQA Prediction Result**: A response for one selected video containing selected identity, predicted assessment, available ground truth, and compact temporal fusion model identity.
- **Sampled Frame Cache Entry**: A reusable frame-sampling output for one video and sampling contract. Key attributes include video identity, sampling contract identity, cache enabled state, and compatibility metadata.
- **VideoMAE Feature Cache Entry**: A reusable feature output for one video and feature contract. Key attributes include video identity, feature extractor identity, preprocessing contract identity, model or artifact version, and compatibility metadata.

## Architecture And Data Impact

- **Touched Surfaces**: Extend the existing trick AQA serving surface in `packages/aitraf-api`; extend or reuse runtime frame sampling, VideoMAE feature extraction, and cache helpers from `packages/aitraf-core`; update `packages/aitraf-train` model configuration, evaluation, registration, or artifact metadata only if the temporal fusion model is not already ready for serving.
- **Shared Helpers To Add Or Extend**: Extend reusable helpers for model-kind mapping, artifact compatibility checks, task/model validation, prediction response shaping, manifest-row selection, frame sampling, VideoMAE feature extraction, model caching, frame cache contract checks, and feature cache contract checks.
- **Data Or Artifact Impact**: Requires current trick AQA demo/test manifest rows, corresponding video assets, a temporal fusion model artifact for trick AQA, packaged metadata needed to preprocess inputs and decode assessment outputs, and cache metadata sufficient to prove cached frames or features match the requested contract.
- **Reproducibility Inputs**: Current trick AQA manifest, selected video assets, temporal fusion model reference, required environment secrets outside committed files, model registration metadata, frame-sampling cache flag, VideoMAE feature cache configuration, cache compatibility metadata, and validation commands or reports showing the model can serve trick AQA predictions.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of successful trick AQA demo prediction responses identify temporal fusion as the model kind.
- **SC-002**: An approved app can complete the trick AQA demo prediction flow for at least one valid current demo video without manual backend intervention.
- **SC-003**: 100% of automated invalid-selection cases return explicit failures and no prediction value.
- **SC-004**: 100% of tested missing temporal fusion configuration or artifact cases return explicit failures and do not use another model's output.
- **SC-005**: Existing validated health, demo-video listing, auth enforcement, and trick classification flows continue to pass after the temporal fusion trick AQA change.
- **SC-006**: Automated or smoke validation confirms temporal fusion training/evaluation and API prediction use shared `aitraf-core` frame sampling and VideoMAE feature functionality.
- **SC-007**: Repeated API predictions for the same valid trick AQA video reuse the cached model instance in 100% of covered cache-enabled cases.
- **SC-008**: Frame-sampling cache validation covers both cache-enabled reuse and cache-disabled recomputation behavior.
- **SC-009**: VideoMAE feature cache validation covers compatible cache reuse and incompatible-contract recomputation or rejection behavior.

## Assumptions

- Trick AQA refers to the repository's existing assessment/score prediction surface used by the demo API.
- The temporal fusion model is the intended serving model for trick AQA, not an optional alternative selected by callers.
- The frontend needs compact model identity for display but does not need internal registry paths, credentials, or training details.
- The feature does not add arbitrary video upload, retraining from the API, new demo-video selection semantics, or public unauthenticated prediction access.
- Missing model references, incompatible artifacts, schema mismatches, and invalid selected videos are surfaced as explicit errors rather than repaired or silently skipped.
- Frame sampling and VideoMAE feature extraction are shared runtime concerns owned by `aitraf-core` and reused by both `aitraf-train` and `aitraf-api`.
- Model caching is enabled for API runtime use; frame-sampling cache reuse is controlled by an explicit flag; VideoMAE feature cache reuse depends on matching feature-contract metadata.
