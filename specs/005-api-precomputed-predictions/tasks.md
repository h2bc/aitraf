# Tasks: AITRAF API Precomputed Predictions

**Input**: Design documents from `/specs/005-api-precomputed-predictions/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/openapi.yaml](./contracts/openapi.yaml), [quickstart.md](./quickstart.md)

**Validation**: A small API endpoint test suite and Docker build are required. Real MLflow run smoke validation is completed only when intentionally run with supplied prediction run IDs.

**Organization**: Tasks are grouped by user story so each story can be implemented and validated independently.

## Phase 1: Setup

**Purpose**: Prepare package config, env docs, and API image direction for the simplified serving path.

- [X] T001 Update root `.env.example` to remove API-specific live model URI variables, unused path variables, and prediction artifact path variables, keep shared train/runtime path variables still used by code, and add `AITRAF_CLASSIFICATION_PREDICTIONS_RUN_ID` and `AITRAF_AQA_PREDICTIONS_RUN_ID` placeholders in `.env.example`
- [X] T002 Update `packages/aitraf-api/pyproject.toml` to remove `aitraf-core`, `torch`, and `transformers` from API runtime dependencies while keeping FastAPI, Pydantic, python-dotenv, uvicorn, and MLflow
- [X] T003 Update `uv.lock` after changing `packages/aitraf-api/pyproject.toml`
- [X] T004 Update `packages/aitraf-api/Dockerfile` to use a CPU Python base, remove ffmpeg installation, stop copying `packages/aitraf-core`, and keep only files needed by `aitraf-api`
- [X] T005 [P] Create the new feature package skeleton in `packages/aitraf-api/src/aitraf_api/features/demo_predictions/__init__.py`

---

## Phase 2: Foundational

**Purpose**: Add the train-side full prediction artifacts and shared API-only config/artifact helpers that block user stories.

**Critical**: No user story work should begin until these train/API boundaries are in place.

- [X] T006 Create shared full prediction table helpers in `packages/aitraf-train/src/aitraf_train/tracking/predictions.py` for converting prediction IDs, labels, confidences, and test-set metadata into MLflow table rows
- [X] T007 Keep train-side export changes untested by repo tests per implementation scope; validate through real eval artifacts instead
- [X] T008 Update all classification eval scripts to log full `test_predictions.json` using the shared helper in `packages/aitraf-train/src/aitraf_train/tasks/trick_classifier/video_mae/evaluation.py`, `packages/aitraf-train/src/aitraf_train/tasks/trick_classifier/video_mae_temporal_fusion/evaluation.py`, and `packages/aitraf-train/src/aitraf_train/tasks/trick_classifier/pose_tcn/evaluation.py`
- [X] T009 Update all score/AQA eval scripts to log full `test_predictions.json` using the shared helper in `packages/aitraf-train/src/aitraf_train/tasks/score_prediction/video_mae/evaluation.py`, `packages/aitraf-train/src/aitraf_train/tasks/score_prediction/pose_tcn/evaluation.py`, `packages/aitraf-train/src/aitraf_train/tasks/score_prediction_ordinal/video_mae/evaluation.py`, `packages/aitraf-train/src/aitraf_train/tasks/score_prediction_ordinal/video_mae_temporal_fusion/evaluation.py`, `packages/aitraf-train/src/aitraf_train/tasks/score_prediction_ordinal/pose_tcn/evaluation.py`, `packages/aitraf-train/src/aitraf_train/tasks/score_prediction_binary/video_mae/evaluation.py`, `packages/aitraf-train/src/aitraf_train/tasks/score_prediction_binary/video_mae_temporal_fusion/evaluation.py`, and `packages/aitraf-train/src/aitraf_train/tasks/score_prediction_pairwise/video_mae/evaluation.py`
- [X] T010 Refactor `packages/aitraf-api/src/aitraf_api/config.py` to replace live model and local metadata settings with only `AITRAF_API_TOKEN`, `AITRAF_CLASSIFICATION_PREDICTIONS_RUN_ID`, and `AITRAF_AQA_PREDICTIONS_RUN_ID`
- [X] T011 Create `packages/aitraf-api/src/aitraf_api/features/demo_predictions/artifacts.py` with fixed `test_predictions.json` artifact path constants, artifact source dataclasses, and direct MLflow table artifact reading
- [X] T012 Implement MLflow artifact download helpers in `packages/aitraf-api/src/aitraf_api/features/demo_predictions/artifacts.py` using configured run IDs and fixed artifact path constants, without importing `aitraf-core`, `torch`, or `transformers`
- [X] T013 Remove shared API dependency helpers and keep the prepared demo predictions app-state lookup inside `packages/aitraf-api/src/aitraf_api/features/demo_predictions/route.py`
- [X] T014 Refactor `packages/aitraf-api/src/aitraf_api/schemas.py` to define `DemoPredictionsResponse`, `DemoPrediction`, `TaskPredictions`, `PredictionRow`, and `GroundTruth`, then remove obsolete live inference schemas if no remaining API code uses them

**Checkpoint**: Train eval can emit full prediction artifacts; API config, artifact download/read helpers, and response schemas are ready for story implementation.

---

## Phase 3: User Story 1 - Serve Demo Results Without Inference (Priority: P1) MVP

**Goal**: The API starts without live models, prepares the matched demo predictions response in memory, and serves it from `GET /demo-predictions`.

**Independent Test**: Start the API with fixture full prediction artifacts, call authenticated `GET /demo-predictions`, and verify each returned record includes metadata plus classification and AQA predictions without invoking model inference.

### Validation for User Story 1

- [X] T015 [P] [US1] Add fixture full prediction artifacts for successful classification/AQA matching in `packages/aitraf-api/tests/features/demo_predictions/conftest.py`
- [X] T016 [P] [US1] Add tests for `GET /demo-predictions` success response shape and authentication in `packages/aitraf-api/tests/features/demo_predictions/test_demo_predictions.py`
- [X] T017 [P] [US1] Keep endpoint tests focused on success response shape and authentication; do not add separate artifact/config unit tests

### Implementation for User Story 1

- [X] T018 [US1] Implement startup preparation of the final demo predictions response in `packages/aitraf-api/src/aitraf_api/app.py` by downloading artifacts, matching classification/AQA rows by `video_id`, and storing the response in app state
- [X] T019 [US1] Implement startup loading, matching, and response construction helpers in `packages/aitraf-api/src/aitraf_api/features/demo_predictions/loader.py`
- [X] T020 [US1] Implement authenticated `GET /demo-predictions` in `packages/aitraf-api/src/aitraf_api/features/demo_predictions/route.py`
- [X] T021 [US1] Export the `demo_predictions` router in `packages/aitraf-api/src/aitraf_api/features/demo_predictions/__init__.py`
- [X] T022 [US1] Update route registration in `packages/aitraf-api/src/aitraf_api/features/__init__.py` to include `demo_predictions` and keep `health`
- [X] T023 [US1] Update API test app construction fixtures to provide simplified settings and prepared demo prediction state in `packages/aitraf-api/tests/conftest.py`

**Checkpoint**: User Story 1 is independently functional with fixture full prediction artifacts and no live inference.

---

## Phase 4: User Story 2 - Keep Predictions Tied To MLOps Results (Priority: P2)

**Goal**: Eval/export can produce full prediction artifacts in MLOps and the API downloads externally supplied MLflow run IDs using fixed code-level artifact paths.

**Independent Test**: Configure the API with MLflow run IDs that contain `test_predictions.json`, then verify artifact download/read logic uses fixed artifact paths and never requires artifact path environment variables.

### Validation for User Story 2

- [X] T024 [P] [US2] Document required prediction run IDs and absence of artifact path env vars in `specs/005-api-precomputed-predictions/quickstart.md`
- [X] T025 [P] [US2] Verify supplied MLflow runs contain fixed `test_predictions.json` artifacts through manual/command validation, not repo unit tests
- [X] T026 [P] [US2] Verify classification eval export behavior through supplied MLflow artifact inspection, not repo train tests
- [X] T027 [P] [US2] Verify score/AQA eval export behavior through supplied MLflow artifact inspection, not repo train tests

### Implementation for User Story 2

- [X] T028 [US2] Wire `AITRAF_CLASSIFICATION_PREDICTIONS_RUN_ID` and `AITRAF_AQA_PREDICTIONS_RUN_ID` through `packages/aitraf-api/src/aitraf_api/config.py` into artifact sources used by `packages/aitraf-api/src/aitraf_api/app.py`
- [X] T029 [US2] Ensure `packages/aitraf-api/src/aitraf_api/features/demo_predictions/artifacts.py` fails loudly for missing run IDs, missing downloaded files, and unreadable JSON files
- [X] T030 [US2] Record externally supplied classification and AQA MLflow run IDs that contain full `test_predictions.json` artifacts in `specs/005-api-precomputed-predictions/quickstart.md`
- [X] T031 [US2] Verify the supplied classification and AQA MLflow runs contain downloadable `test_predictions.json` artifacts with `video_id`, `s3_path`, `person`, `trick`, `execution_score`, and prediction label fields in `specs/005-api-precomputed-predictions/quickstart.md`
- [X] T033 [US2] Verify generated prediction artifacts are not added to source control and update `.gitignore` only if implementation creates a new local artifact cache path in `.gitignore`

**Checkpoint**: User Story 2 is independently functional with explicit run ID config, fixed artifact path constants, and full prediction artifacts in MLOps.

---

## Phase 5: User Story 3 - Remove Unsupported Online Inference Behavior (Priority: P3)

**Goal**: The API surface and runtime dependencies no longer expose or support arbitrary live inference.

**Independent Test**: Call old per-video inference routes and verify they are unavailable; inspect dependency/image config and verify live model runtime dependencies are absent from the API runtime.

### Validation for User Story 3

- [X] T034 [P] [US3] Remove old live inference route tests instead of keeping artificial 404 assertions for deleted routes
- [X] T035 [P] [US3] Verify by source/dependency inspection that `aitraf_api.app` no longer imports `aitraf_core`, `torch`, or `transformers`, without adding a legacy/runtime dependency test file

### Implementation for User Story 3

- [X] T036 [US3] Remove trick classification route registration and delete obsolete code under `packages/aitraf-api/src/aitraf_api/features/trick_classification/`
- [X] T037 [US3] Remove trick AQA route registration and delete obsolete code under `packages/aitraf-api/src/aitraf_api/features/trick_assessment/`
- [X] T038 [US3] Remove obsolete live inference helper code in `packages/aitraf-api/src/aitraf_api/video_loading.py`
- [X] T039 [US3] Remove obsolete live model state, imports, and preprocessing setup from `packages/aitraf-api/src/aitraf_api/app.py`
- [X] T040 [US3] Delete obsolete shared API dependency module after removing live model dependency getters
- [X] T041 [US3] Delete or rewrite old live inference tests in `packages/aitraf-api/tests/features/trick_classification/test_trick_classification.py` and `packages/aitraf-api/tests/features/trick_assessment/test_trick_assessment.py`

**Checkpoint**: User Story 3 is independently functional and the API no longer presents live inference behavior.

---

## Phase 6: Polish & Cross-Cutting Validation

**Purpose**: Documentation, command validation, Docker validation, and final cleanup across all stories.

- [X] T042 [P] Update `packages/aitraf-api/README.md` to document `GET /demo-predictions`, prediction run ID environment variables, fixed `test_predictions.json` artifact path constants, removed live inference config, and the new validation run IDs
- [X] T043 [P] Update `specs/005-api-precomputed-predictions/quickstart.md` with the externally supplied classification and AQA run IDs that contain full `test_predictions.json` artifacts, when available
- [X] T044 Confirm no `aitraf-train` repo tests are required or added for this API simplification scope
- [X] T045 Run `uv run --package aitraf-api pytest packages/aitraf-api/tests/features/demo_predictions` and record the result in `specs/005-api-precomputed-predictions/quickstart.md`
- [X] T046 Build the API image with `docker build -f packages/aitraf-api/Dockerfile -t aitraf-api:precomputed .` and record the result in `specs/005-api-precomputed-predictions/quickstart.md`
- [X] T047 Run the Docker image with supplied classification and AQA run IDs from `specs/005-api-precomputed-predictions/quickstart.md`, let the container download MLflow `test_predictions.json` artifacts at startup, then verify `/health` and authenticated `GET /demo-predictions`
- [X] T048 Verify `packages/aitraf-api/Dockerfile`, `packages/aitraf-api/pyproject.toml`, API source, and `uv tree --package aitraf-api --no-dev` do not include API runtime dependencies on `aitraf-core`, `torch`, `transformers`, CUDA, or ffmpeg
- [X] T049 Remove stale generated Python cache files under `packages/aitraf-api/src/aitraf_api/__pycache__/`, `packages/aitraf-api/tests/**/__pycache__/`, and `packages/aitraf-train/**/__pycache__/` if they remain tracked or appear in review output

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 Setup**: No dependencies.
- **Phase 2 Foundational**: Depends on Phase 1 and blocks all user stories.
- **Phase 3 US1**: Depends on Phase 2. Delivers fixture-based MVP API behavior.
- **Phase 4 US2**: Depends on Phase 2 and consumes externally supplied full prediction artifacts in MLOps.
- **Phase 5 US3**: Depends on Phase 2 and should be completed before final Docker validation.
- **Phase 6 Polish**: Depends on US1, US2, and US3, with real MLflow Docker smoke validation depending on externally supplied run IDs.

### User Story Dependencies

- **US1 Serve Demo Results Without Inference**: Starts after foundational helpers. MVP with fixture full prediction artifacts.
- **US2 Keep Predictions Tied To MLOps Results**: Starts after foundational train/API artifact helper structure. Real Docker smoke validation requires externally supplied run IDs.
- **US3 Remove Unsupported Online Inference Behavior**: Starts after foundational route/config decisions. Can be implemented in parallel with US1/US2 if conflicts in `app.py` and `features/__init__.py` are coordinated.

### Blocking Notes

- T006-T014 must complete before route/service implementation.
- T018-T023 must complete before the MVP can be tested through `/demo-predictions`.
- T030-T031 require externally supplied run IDs before real Docker smoke validation because current inspected runs lack full prediction artifacts.
- T036-T041 must complete before claiming no legacy inference surface remains.
- T044-T048 are required completion gates.

---

## Parallel Execution Examples

### User Story 1

```text
Task: "T015 Add fixture prepared demo prediction response in packages/aitraf-api/tests/features/demo_predictions/conftest.py"
Task: "T016 Add GET /demo-predictions response and auth tests in packages/aitraf-api/tests/features/demo_predictions/test_demo_predictions.py"
Task: "T017 Do not add separate artifact/config unit tests for this API-focused scope"
```

### User Story 2

```text
Task: "T024 Document run ID config in specs/005-api-precomputed-predictions/quickstart.md"
Task: "T025 Verify supplied MLflow artifacts by command/manual inspection"
Task: "T026 Verify classification export through supplied MLflow artifact inspection"
Task: "T027 Verify AQA export through supplied MLflow artifact inspection"
```

### User Story 3

```text
Task: "T034 Remove old live inference route tests"
Task: "T035 Verify dependency removal by source/dependency inspection"
```

### Polish

```text
Task: "T042 Update packages/aitraf-api/README.md"
Task: "T043 Update specs/005-api-precomputed-predictions/quickstart.md with supplied run IDs"
```

---

## Implementation Strategy

### MVP First

1. Complete Phase 1 and Phase 2.
2. Complete Phase 3 only.
3. Validate with fixture full prediction artifacts using `uv run --package aitraf-api pytest packages/aitraf-api/tests`.
4. Confirm authenticated `GET /demo-predictions` returns cached demo prediction records without live inference.

### Full Feature

1. Complete MVP.
2. Complete US2 to log full `test_predictions.json` artifacts from all `aitraf-train` evaluation scripts and wire API downloads for externally supplied MLflow run IDs.
3. Complete US3 to remove live inference routes and dependencies.
4. Complete Phase 6, including train tests, API tests, Docker build, and real Docker smoke validation once full-prediction run IDs are supplied.

### Constitution Alignment

- No fallback to live inference is allowed.
- No compatibility route for old `/inference/*` behavior is allowed.
- Prediction artifact readers must accept the single required
  `test_predictions.json` schema and reject alternate shapes instead of
  normalizing multiple possible formats.
- API-owned MLflow helpers must not import `aitraf-core` model loading.
- Full prediction artifacts are generated by `aitraf-train` and must remain outside source control.
