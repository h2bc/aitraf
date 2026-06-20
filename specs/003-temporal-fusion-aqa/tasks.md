# Tasks: Temporal Fusion Trick AQA

**Input**: Design documents from `/specs/003-temporal-fusion-aqa/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Validation**: Validation tasks are REQUIRED. Add automated tests where practical and always add command-level smoke validation for pipeline behavior.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **API package**: `packages/aitraf-api/src/aitraf_api/`, `packages/aitraf-api/tests/`
- **Core runtime package**: `packages/aitraf-core/src/aitraf_core/`
- **Train package configs**: `packages/aitraf-train/configs/`
- **Train package entrypoints**: `packages/aitraf-train/scripts/`
- **Train package code**: `packages/aitraf-train/src/aitraf_train/`
- **Validation**: package-local `tests/`, `tests/unit/`, `tests/integration/`, `tests/smoke/`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the shared module scaffolding and test locations needed by later user stories.

- [X] T001 Create `aitraf_core.pre_processing` package scaffold in `packages/aitraf-core/src/aitraf_core/pre_processing/__init__.py`
- [X] T002 Create VideoMAE pre-processing package scaffold in `packages/aitraf-core/src/aitraf_core/pre_processing/models/__init__.py`
- [X] T003 [P] Create core test package scaffold in `packages/aitraf-core/tests/test_pre_processing_cache.py`
- [X] T004 [P] Create temporal-fusion inference test scaffold in `packages/aitraf-core/tests/test_temporal_fusion_inference.py`
- [X] T005 [P] Create train reuse test scaffold in `packages/aitraf-train/tests/test_video_mae_temporal_fusion_reuse.py`
- [X] T006 [P] Create API trick AQA temporal-fusion test scaffold in `packages/aitraf-api/tests/features/trick_assessment/test_temporal_fusion.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Define the shared contracts and compatibility rules that block all story implementation.

**CRITICAL**: No user story work can begin until this phase is complete.

- [X] T007 Define sampled-frame and VideoMAE feature cache contract dataclasses in `packages/aitraf-core/src/aitraf_core/pre_processing/cache.py`
- [X] T008 Keep cache contract failures simple and explicit in `packages/aitraf-core/src/aitraf_core/pre_processing/cache.py`
- [X] T009 Add temporal-fusion feature/result dataclasses for pre-processing outputs in `packages/aitraf-core/src/aitraf_core/pre_processing/models/video_mae.py`
- [X] T010 Keep temporal-fusion feature sample processing simple in `packages/aitraf-core/src/aitraf_core/processing/models/video_mae_temporal_fusion.py`
- [X] T011 [P] Add label helper compatibility exports in `packages/aitraf-core/src/aitraf_core/processing/labels.py`
- [X] T012 Update core processing package exports for new labels and temporal-fusion helpers in `packages/aitraf-core/src/aitraf_core/processing/__init__.py`
- [X] T013 Update core pre-processing package exports in `packages/aitraf-core/src/aitraf_core/pre_processing/__init__.py`
- [X] T014 Update core inference package exports for temporal-fusion model helpers in `packages/aitraf-core/src/aitraf_core/inference/__init__.py`

**Checkpoint**: Foundation ready - user story implementation can now begin in priority order or in parallel where noted.

---

## Phase 3: User Story 1 - Predict Trick AQA With Temporal Fusion (Priority: P1) MVP

**Goal**: The trick AQA API prediction flow uses the configured temporal-fusion model and returns a decoded assessment result with temporal-fusion model identity.

**Independent Test**: Select a valid trick AQA demo video and confirm the API response contains the selected video id, decoded prediction, ground truth, and temporal-fusion model kind without changing trick classification behavior.

### Validation for User Story 1

- [X] T015 [P] [US1] Add core temporal-fusion prediction unit tests for logits, confidence, and decode errors in `packages/aitraf-core/tests/test_temporal_fusion_inference.py`
- [X] T016 [P] [US1] Add API success and invalid-selection tests for temporal-fusion trick AQA in `packages/aitraf-api/tests/features/trick_assessment/test_temporal_fusion.py`
- [X] T017 [P] [US1] Add API no-fallback test asserting AQA model kind is temporal fusion in `packages/aitraf-api/tests/features/trick_assessment/test_temporal_fusion.py`

### Implementation for User Story 1

- [X] T018 [P] [US1] Implement temporal-fusion logits and decoded prediction helpers in `packages/aitraf-core/src/aitraf_core/inference/models/video_mae_temporal_fusion.py`
- [X] T019 [P] [US1] Keep temporal-fusion inference metadata handling simple in `packages/aitraf-core/src/aitraf_core/inference/models/video_mae_temporal_fusion.py`
- [X] T020 [US1] Extend API registered model settings with temporal-fusion AQA kind and cache-related fields in `packages/aitraf-api/src/aitraf_api/config.py`
- [X] T021 [US1] Update API prediction orchestration to call temporal-fusion `pre_processing`, `processing`, and `predict` helpers for AQA in `packages/aitraf-api/src/aitraf_api/prediction.py`
- [X] T022 [US1] Update trick assessment service to pass cache flags and use temporal-fusion prediction orchestration in `packages/aitraf-api/src/aitraf_api/features/trick_assessment/service.py`
- [X] T023 [US1] Update trick assessment route to accept optional `cache_frames` and `cache_video_features` query flags in `packages/aitraf-api/src/aitraf_api/features/trick_assessment/route.py`
- [X] T024 [US1] Update API response schema constraints for temporal-fusion model identity in `packages/aitraf-api/src/aitraf_api/schemas.py`
- [X] T025 [US1] Update API test fixtures to stub temporal-fusion model metadata and prediction calls in `packages/aitraf-api/tests/conftest.py`
- [X] T026 [US1] Document the API temporal-fusion smoke request and expected response in `packages/aitraf-api/README.md`

**Checkpoint**: User Story 1 is independently functional when the focused API tests pass and the documented trick AQA request returns temporal-fusion model identity.

---

## Phase 4: User Story 2 - Reuse Core Processing Across Training And Serving (Priority: P1)

**Goal**: Temporal-fusion data pipelines, train/eval flows, and API inference use shared `aitraf-core` pre-processing and processing helpers instead of duplicating frame sampling and VideoMAE feature extraction.

**Independent Test**: Validate that `aitraf-train` temporal-fusion data/training/evaluation and API inference import and call shared `aitraf-core` helpers rather than task-local copies.

### Validation for User Story 2

- [X] T027 [P] [US2] Add core VideoMAE pre-processing unit tests for segmented frame processing and feature extraction dependency injection in `packages/aitraf-core/tests/test_video_mae_pre_processing.py`
- [X] T028 [P] [US2] Add train data-op reuse tests that patch `aitraf_core.pre_processing` helpers in `packages/aitraf-train/tests/test_video_mae_temporal_fusion_reuse.py`
- [X] T029 [P] [US2] Add temporal-fusion train/eval processing reuse tests in `packages/aitraf-train/tests/test_video_mae_temporal_fusion_reuse.py`

### Implementation for User Story 2

- [X] T030 [P] [US2] Move segmented frame-to-pixel-values preparation from train data ops into `packages/aitraf-core/src/aitraf_core/pre_processing/models/video_mae.py`
- [X] T031 [P] [US2] Move VideoMAE batch feature extraction from train data ops into `packages/aitraf-core/src/aitraf_core/pre_processing/models/video_mae.py`
- [X] T032 [US2] Refactor `packages/aitraf-train/src/aitraf_train/data_ops/video_mae_feature_extraction.py` to orchestrate logging, batching, and persistence through `aitraf_core.pre_processing`
- [X] T033 [US2] Keep temporal-fusion feature sample processing in core without extra wrapper objects in `packages/aitraf-core/src/aitraf_core/processing/models/video_mae_temporal_fusion.py`
- [X] T034 [US2] Confirm ordinal temporal-fusion training uses the core processing helper in `packages/aitraf-train/src/aitraf_train/tasks/score_prediction_ordinal/video_mae_temporal_fusion/training.py`
- [X] T035 [US2] Confirm ordinal temporal-fusion evaluation uses the core processing helper in `packages/aitraf-train/src/aitraf_train/tasks/score_prediction_ordinal/video_mae_temporal_fusion/evaluation.py`
- [X] T036 [US2] Confirm binary temporal-fusion training/evaluation imports use the core helper in `packages/aitraf-train/src/aitraf_train/tasks/score_prediction_binary/video_mae_temporal_fusion/training.py`
- [X] T037 [US2] Confirm trick-classification temporal-fusion training/evaluation imports use the core helper in `packages/aitraf-train/src/aitraf_train/tasks/trick_classifier/video_mae_temporal_fusion/training.py`
- [X] T038 [US2] Leave data ops pipeline config mapping unchanged because no new required pre-processing fields were added in `packages/aitraf-train/scripts/data_ops_pipeline.py`
- [X] T039 [US2] Add command-level reuse validation notes for data ops, train, and eval in `packages/aitraf-train/README.md`

**Checkpoint**: User Story 2 is independently functional when train/data-op tests prove shared core helper usage and the data-op smoke command still produces VideoMAE feature artifacts.

---

## Phase 5: User Story 3 - Cache Reusable Temporal Fusion Inputs (Priority: P2)

**Goal**: Repeated trick AQA predictions reuse cached model instances, sampled frames, and VideoMAE features when enabled and contract-compatible.

**Independent Test**: Make repeated requests for the same valid trick AQA video with caching enabled and verify the same result is returned while model loading, frame sampling, and feature extraction are not repeated.

### Validation for User Story 3

- [X] T040 [P] [US3] Add cache contract tests for compatible reuse, incompatible rejection, and disabled-cache recomputation in `packages/aitraf-core/tests/test_pre_processing_cache.py`
- [ ] T041 [P] [US3] Add API repeated-request cache tests for model, frame, and VideoMAE feature reuse in `packages/aitraf-api/tests/features/trick_assessment/test_temporal_fusion.py`
- [ ] T042 [P] [US3] Add data-op cache metadata read/write tests in `packages/aitraf-train/tests/test_video_mae_temporal_fusion_reuse.py`

### Implementation for User Story 3

- [X] T043 [P] [US3] Implement sampled-frame cache read/write helpers in `packages/aitraf-core/src/aitraf_core/pre_processing/cache.py`
- [X] T044 [P] [US3] Implement VideoMAE feature cache read/write helpers with contract metadata in `packages/aitraf-core/src/aitraf_core/pre_processing/cache.py`
- [ ] T045 [US3] Integrate sampled-frame cache flags into VideoMAE pre-processing in `packages/aitraf-core/src/aitraf_core/pre_processing/models/video_mae.py`
- [ ] T046 [US3] Integrate VideoMAE feature cache flags into VideoMAE pre-processing in `packages/aitraf-core/src/aitraf_core/pre_processing/models/video_mae.py`
- [X] T047 [US3] Extend API settings with frame cache and feature cache directories/default flags in `packages/aitraf-api/src/aitraf_api/config.py`
- [X] T048 [US3] Verify loaded temporal-fusion model cache behavior and cache key correctness in `packages/aitraf-api/src/aitraf_api/prediction.py`
- [ ] T049 [US3] Add Hydra config fields for sampled-frame cache and feature cache contract metadata in `packages/aitraf-train/configs/data_ops.yaml`
- [ ] T050 [US3] Persist cache contract metadata from the train data op in `packages/aitraf-train/src/aitraf_train/data_ops/video_mae_feature_extraction.py`
- [X] T051 [US3] Document cache flags, cache paths, and invalidation behavior in `packages/aitraf-core/README.md`

**Checkpoint**: User Story 3 is independently functional when compatible repeated requests reuse caches and incompatible cache entries are rejected or explicitly recomputed.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Complete validation, documentation, and cleanup across all stories.

- [X] T052 [P] Update feature quickstart with final command names and cache env/config names in `specs/003-temporal-fusion-aqa/quickstart.md`
- [X] T053 [P] Update core pipeline contract if implementation names changed in `specs/003-temporal-fusion-aqa/contracts/core-pipeline.md`
- [X] T054 [P] Update API contract if final query flag or response schema names changed in `specs/003-temporal-fusion-aqa/contracts/openapi.yaml`
- [X] T055 Run core, train, and API automated tests with `uv run pytest packages/aitraf-core packages/aitraf-train packages/aitraf-api` and record outcome in `specs/003-temporal-fusion-aqa/quickstart.md`
- [ ] T056 Run data-op smoke command from `specs/003-temporal-fusion-aqa/quickstart.md` and record cache artifact outcome in `specs/003-temporal-fusion-aqa/quickstart.md`
- [ ] T057 Run API smoke request from `specs/003-temporal-fusion-aqa/quickstart.md` and record temporal-fusion response outcome in `specs/003-temporal-fusion-aqa/quickstart.md`
- [X] T058 Remove any duplicated temporal-fusion frame sampling or VideoMAE feature extraction logic left in `packages/aitraf-train/src/aitraf_train/data_ops/video_mae_feature_extraction.py`
- [X] T059 Remove any API-local temporal-fusion preprocessing or decoding constants that duplicate artifact/core metadata in `packages/aitraf-api/src/aitraf_api/prediction.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion - blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational completion and is the MVP.
- **User Story 2 (Phase 4)**: Depends on Foundational completion; can run in parallel with US1 after shared contracts are stable, but should be reconciled before final API smoke validation.
- **User Story 3 (Phase 5)**: Depends on Foundational completion and benefits from US1/US2 integration points.
- **Polish (Phase 6)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **US1 - Predict Trick AQA With Temporal Fusion (P1)**: Start after Phase 2. Delivers the MVP API capability.
- **US2 - Reuse Core Processing Across Training And Serving (P1)**: Start after Phase 2. Independent value is reuse validation across train/API, but final API behavior should consume the same helpers as US1.
- **US3 - Cache Reusable Temporal Fusion Inputs (P2)**: Start after Phase 2. Depends on the pre-processing contracts created for US1/US2.

### Within Each User Story

- Validation tasks are listed before implementation tasks.
- Core helper tasks precede train/API integration tasks.
- Config and schema tasks precede endpoint/service smoke validation.
- Cache contract tasks precede cache integration tasks.

### Parallel Opportunities

- Setup scaffolds T003-T006 can run in parallel.
- Foundational exports T011 and initial contract work T007-T010 can be split by file, then reconciled through T012-T014.
- US1 validation tasks T015-T017 can run in parallel before implementation.
- US2 validation tasks T027-T029 can run in parallel before implementation.
- US2 core pre-processing tasks T030-T031 can run in parallel after foundational contracts.
- US3 validation tasks T040-T042 can run in parallel before implementation.
- US3 cache helper tasks T043-T044 can run in parallel before cache integration.
- Polish documentation tasks T052-T054 can run in parallel.

---

## Parallel Example: User Story 1

```bash
Task: "T015 [P] [US1] Add core temporal-fusion prediction unit tests in packages/aitraf-core/tests/test_temporal_fusion_inference.py"
Task: "T016 [P] [US1] Add API success and invalid-selection tests in packages/aitraf-api/tests/features/trick_assessment/test_temporal_fusion.py"
Task: "T017 [P] [US1] Add API no-fallback test in packages/aitraf-api/tests/features/trick_assessment/test_temporal_fusion.py"
```

## Parallel Example: User Story 2

```bash
Task: "T027 [P] [US2] Add core VideoMAE pre-processing unit tests in packages/aitraf-core/tests/test_video_mae_pre_processing.py"
Task: "T028 [P] [US2] Add train data-op reuse tests in packages/aitraf-train/tests/test_video_mae_temporal_fusion_reuse.py"
Task: "T030 [P] [US2] Move segmented frame-to-pixel-values preparation into packages/aitraf-core/src/aitraf_core/pre_processing/models/video_mae.py"
```

## Parallel Example: User Story 3

```bash
Task: "T040 [P] [US3] Add cache contract tests in packages/aitraf-core/tests/test_pre_processing_cache.py"
Task: "T041 [P] [US3] Add API repeated-request cache tests in packages/aitraf-api/tests/features/trick_assessment/test_temporal_fusion.py"
Task: "T043 [P] [US3] Implement sampled-frame cache helpers in packages/aitraf-core/src/aitraf_core/pre_processing/cache.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 setup.
2. Complete Phase 2 foundational contracts and exports.
3. Complete Phase 3 US1 temporal-fusion API prediction.
4. Validate US1 with focused core/API tests and one trick AQA API smoke request.

### Incremental Delivery

1. Deliver US1 so the API can predict trick AQA with temporal fusion.
2. Deliver US2 to remove duplicated train/API pre-processing and prove shared core reuse.
3. Deliver US3 to add explicit model, frame, and VideoMAE feature cache behavior.
4. Complete Phase 6 smoke validation and documentation updates.

### Parallel Team Strategy

With multiple developers:

1. Complete Setup and Foundational phases together.
2. Assign US1 API/inference work to one developer.
3. Assign US2 train/core reuse work to a second developer.
4. Assign US3 cache behavior to a third developer after cache contracts stabilize.
5. Reconcile through shared core exports and run full validation.

---

## Notes

- `[P]` tasks use different files or can be prepared without depending on incomplete implementation details.
- Story labels map to the user stories in `specs/003-temporal-fusion-aqa/spec.md`.
- All invalid states must fail explicitly or recompute only when a recompute path is configured.
- Do not import `aitraf-train` from `aitraf-api`.
- Do not leave production temporal-fusion behavior in notebooks or local-only scripts.
