# Tasks: Publish Demo Assets

**Input**: Design documents from `/specs/007-publish-demo-assets/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Validation**: Automated tests and command-level smoke validation are required
by the feature specification.

**Organization**: Tasks are grouped by user story so stable public serving,
subset-only publication, and safe idempotent restarts can be implemented and
verified as distinct increments.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it changes different files and does not
  depend on an incomplete task.
- **[Story]**: Maps the task to US1, US2, or US3 from the specification.
- Every task names the exact file or directory it changes.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the single required public-bucket configuration and
document the runtime boundary before publication logic is introduced.

- [ ] T001 Add required `public_asset_bucket: str` loading from `AITRAF_PUBLIC_ASSET_BUCKET` and strict source/public bucket validation in `packages/aitraf-api/src/aitraf_api/config.py`
- [ ] T002 [P] Add `AITRAF_PUBLIC_ASSET_BUCKET` beside the existing shared endpoint, credentials, and private `AWS_BUCKET` in `.env.example`
- [ ] T003 [P] Update API settings fixtures and missing/invalid public-bucket coverage in `packages/aitraf-api/tests/test_config.py` and `packages/aitraf-api/tests/features/conftest.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Provide strict storage and selected-asset primitives used by every
story.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T004 Add exact object metadata inspection and same-client server-side bucket copy primitives with explicit not-found versus access/error behavior in `packages/aitraf-core/src/aitraf_core/storage/s3.py` and exports in `packages/aitraf-core/src/aitraf_core/storage/__init__.py`
- [ ] T005 [P] Add core storage tests for metadata inspection, missing-object classification, copy arguments, and propagated permission/network failures in `packages/aitraf-core/tests/test_storage_s3.py`
- [ ] T006 Define immutable selected/public demo asset types plus strict `video_id`, source URI, public-key, percent-encoded URL, provenance, deduplication, and collision helpers in `packages/aitraf-api/src/aitraf_api/features/demo_predictions/assets.py`
- [ ] T007 [P] Add pure-helper tests for `videos/<video_id>`, `thumbnails/<video-stem>.jpg`, endpoint slash normalization, URL encoding, cross-bucket rejection, invalid IDs, duplicate collapse, and conflicting mappings in `packages/aitraf-api/tests/features/demo_predictions/test_assets.py`

**Checkpoint**: Configuration, storage primitives, and deterministic asset
contracts are ready for story implementation.

---

## Phase 3: User Story 1 - Serve Stable Public Demo Assets (Priority: P1) 🎯 MVP

**Goal**: Publish missing selected videos and thumbnails before readiness and
serve constant queryless public URLs with no API presigning behavior.

**Independent Test**: Start with one valid matched prediction whose public video
and thumbnail are missing, verify both become public, then call
`GET /demo-predictions` twice and receive identical anonymously reachable URLs
without any signing query parameters.

### Validation for User Story 1 ⚠️

- [ ] T008 [P] [US1] Replace video/presigner tests with missing-video copy, existing-video reuse, public URL, and storage failure coverage in `packages/aitraf-api/tests/features/demo_predictions/test_videos.py`
- [ ] T009 [P] [US1] Refactor thumbnail tests for public-bucket keys, missing-only source download/generation/upload, existing-thumbnail reuse, and explicit FFmpeg/upload failure in `packages/aitraf-api/tests/features/demo_predictions/test_thumbnails.py`
- [ ] T010 [P] [US1] Replace signed-link response tests with exact constant public video/thumbnail URLs across repeated requests and zero request-time storage behavior in `packages/aitraf-api/tests/features/demo_predictions/test_demo_predictions.py`

### Implementation for User Story 1

- [ ] T011 [US1] Implement public video inspection, provenance validation, and missing-only private-to-public copy in `packages/aitraf-api/src/aitraf_api/features/demo_predictions/assets.py`
- [ ] T012 [US1] Refactor thumbnail preparation to generate and upload only missing public thumbnails and return provenance-validated public asset results in `packages/aitraf-api/src/aitraf_api/features/demo_predictions/thumbnails.py`
- [ ] T013 [US1] Replace the old startup video-download/thumbnail/presigner sequence with matched-asset preparation and prepared URL rows in `packages/aitraf-api/src/aitraf_api/app.py`
- [ ] T014 [US1] Remove the presigner callback from response construction and map required prepared `video_url` and `thumbnail_url` fields directly in `packages/aitraf-api/src/aitraf_api/features/demo_predictions/service.py` and `packages/aitraf-api/src/aitraf_api/features/demo_predictions/route.py`
- [ ] T015 [US1] Delete `VIDEO_URL_EXPIRATION_SECONDS`, `AssetUrlPresigner`, `create_asset_url_presigner`, presigning imports, and obsolete download orchestration from `packages/aitraf-api/src/aitraf_api/features/demo_predictions/videos.py`, removing the file if no focused behavior remains

**Checkpoint**: The API serves one successfully published demo asset through
stable public URLs and contains no temporary-link serving path.

---

## Phase 4: User Story 2 - Publish Only The Selected Subset (Priority: P2)

**Goal**: Treat matched classification/AQA prediction rows as the exclusive
publication allowlist and avoid touching unrelated source media.

**Independent Test**: Provide matched, classification-only, AQA-only, duplicate,
and unrelated source rows; verify storage operations occur exactly once for each
unique matched asset and never for unmatched or unrelated videos.

### Validation for User Story 2 ⚠️

- [ ] T016 [P] [US2] Add application-factory integration coverage for matched-subset-only publication, unmatched-row exclusion, duplicate collapse, and added-row-only writes in `packages/aitraf-api/tests/test_app.py`
- [ ] T017 [P] [US2] Add service coverage proving only prepared matched rows can produce responses and raw artifact thumbnail/source paths are never returned in `packages/aitraf-api/tests/features/demo_predictions/test_demo_predictions.py`

### Implementation for User Story 2

- [ ] T018 [US2] Integrate selected-asset derivation immediately after `match_prediction_rows` and perform one deduplicated preparation pass in `packages/aitraf-api/src/aitraf_api/app.py`
- [ ] T019 [US2] Require the single prepared-row media shape and remove `thumbnail_s3_path` mutation or artifact-provided thumbnail selection from `packages/aitraf-api/src/aitraf_api/features/demo_predictions/thumbnails.py` and `packages/aitraf-api/src/aitraf_api/features/demo_predictions/service.py`

**Checkpoint**: Unmatched and unrelated source objects are outside the
publication path while User Story 1 behavior remains intact.

---

## Phase 5: User Story 3 - Restart Safely And Predictably (Priority: P3)

**Goal**: Make repeated startup zero-write and fail before readiness on missing
sources, permissions, races, or existing-object provenance conflicts without
overwriting or deleting production objects.

**Independent Test**: Run preparation twice with identical inputs and assert the
second pass performs zero copies, downloads, or uploads; then inject each
required failure and verify application construction fails with the affected
asset while existing and unselected objects remain unchanged.

### Validation for User Story 3 ⚠️

- [ ] T020 [P] [US3] Add idempotent second-start, same-source race reuse, missing/mismatched provenance, missing source, denied access, and no-delete assertions in `packages/aitraf-api/tests/features/demo_predictions/test_assets.py`
- [ ] T021 [P] [US3] Add application readiness failure tests for copy, source download, thumbnail generation, upload, and public object conflict errors in `packages/aitraf-api/tests/test_app.py`

### Implementation for User Story 3

- [ ] T022 [US3] Complete provenance metadata creation/comparison and post-race destination reinspection so only the same source identity is reused in `packages/aitraf-api/src/aitraf_api/features/demo_predictions/assets.py`
- [ ] T023 [US3] Add contextual startup errors and publication summary logging for copied, generated, reused, and conflicting assets without fallback or cleanup behavior in `packages/aitraf-api/src/aitraf_api/features/demo_predictions/assets.py` and `packages/aitraf-api/src/aitraf_api/app.py`

**Checkpoint**: All three stories are independently covered and startup cannot
report readiness with missing, conflicting, or partially prepared media.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Remove all superseded API behavior, align documentation, and prove
the real bucket works anonymously and idempotently.

- [ ] T024 [P] Rewrite public demo media configuration, startup behavior, stable URL semantics, and anonymous-access expectations in `packages/aitraf-api/README.md`
- [ ] T025 [P] Update architecture references from private thumbnails and presigned API media to startup-published public assets in `packages/aitraf-core/README.md` and any current root documentation that describes the API media flow
- [ ] T026 Remove every remaining API presigner symbol, expiration constant, app-state field, callback, signed-link fixture/test, compatibility branch, obsolete import, and dead module path under `packages/aitraf-api/`
- [ ] T027 Run `uv run pytest packages/aitraf-core/tests packages/aitraf-api/tests -q` and record the results in `specs/007-publish-demo-assets/validation.md`
- [ ] T028 Run the API presigning-removal `rg` scan from `specs/007-publish-demo-assets/quickstart.md`, verify no API runtime/test/doc matches while the train-only generic core consumer remains valid, and record the result in `specs/007-publish-demo-assets/validation.md`
- [ ] T029 Execute the configured startup and authenticated `/demo-predictions` smoke test, anonymously GET every returned video and thumbnail, restart unchanged to verify identical URLs and zero writes, and record inputs/results without secrets in `specs/007-publish-demo-assets/validation.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; T002 can run alongside T001, and T003
  can begin once the target settings shape is agreed.
- **Foundational (Phase 2)**: Depends on Setup; blocks all stories. Core work
  T004/T005 and API pure-helper work T006/T007 can proceed in parallel pairs.
- **User Story 1 (Phase 3)**: Depends on Foundational and provides the MVP public
  serving path.
- **User Story 2 (Phase 4)**: Depends on the US1 preparation integration because
  it narrows that flow to the authoritative matched subset.
- **User Story 3 (Phase 5)**: Depends on US1 publication behavior and can be
  developed alongside US2 after the US1 startup path exists.
- **Polish (Phase 6)**: Depends on all selected stories; runtime smoke validation
  follows automated tests and removal scanning.

### User Story Dependencies

```text
Setup -> Foundation -> US1 (stable public assets) -> US2 (subset allowlist)
                                             `----> US3 (safe restart)
US1 + US2 + US3 -> Polish and real-environment validation
```

- **US1 (P1)**: First independently deliverable slice after Foundation.
- **US2 (P2)**: Reuses US1 publication but is independently verified through an
  explicit mixed matched/unmatched dataset.
- **US3 (P3)**: Reuses US1 publication but is independently verified through a
  second-run and failure matrix; it does not depend on US2 implementation.

### Within Each User Story

- Write or replace the story's tests before completing its implementation.
- Implement pure derivation and storage helpers before application integration.
- Complete startup integration before route/service response assertions.
- Do not mark a story complete until its independent test passes.

### Parallel Opportunities

- T002 and T003 target different setup files.
- T005 and T007 test independent core/API foundational surfaces.
- T008, T009, and T010 define independent US1 video, thumbnail, and response
  expectations.
- T016 and T017 cover separate US2 application/service boundaries.
- T020 and T021 cover separate US3 preparation/application boundaries.
- T024 and T025 update separate documentation surfaces.

---

## Parallel Example: User Story 1

```text
Task T008: Replace video/presigner tests in test_videos.py
Task T009: Refactor public thumbnail tests in test_thumbnails.py
Task T010: Replace signed response tests in test_demo_predictions.py
```

After those contracts are established, implement T011 and T012 in their focused
modules, then integrate them sequentially through T013-T015.

## Parallel Example: User Story 2

```text
Task T016: Add matched-subset application integration tests in test_app.py
Task T017: Add prepared matched-row response tests in test_demo_predictions.py
```

## Parallel Example: User Story 3

```text
Task T020: Add idempotence/provenance tests in test_assets.py
Task T021: Add readiness failure tests in test_app.py
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete T001-T007 for strict configuration and asset/storage foundations.
2. Complete T008-T015 to publish one selected video/thumbnail and serve stable
   public URLs.
3. Stop and run the US1 independent test before expanding subset and restart
   guarantees.
4. Do not ship the old presigned behavior as a fallback during the MVP step.

### Incremental Delivery

1. Setup + Foundation establish one explicit public-asset representation.
2. US1 replaces temporary links with stable public media.
3. US2 proves the matched subset is the complete publication allowlist.
4. US3 adds provenance and restart/failure guarantees without a second path.
5. Polish deletes remaining legacy references and validates the real public
   bucket anonymously.

### Parallel Team Strategy

After Foundation, one developer can prepare the US1 storage/thumbnail modules
while another prepares response tests. Once US1 integrates, US2 subset tests and
US3 restart/failure tests can proceed concurrently because they exercise
different acceptance dimensions.

## Notes

- `[P]` tasks target different files or independent boundaries.
- Story labels map directly to the three prioritized specification stories.
- API presigning is deleted, not deprecated, aliased, feature-flagged, or kept as
  fallback behavior.
- The generic core `presign_s3_uri` remains only because offline training metrics
  still consume it; it must not be referenced from `packages/aitraf-api`.
- Public objects are additive and immutable under this startup workflow; asset
  retention and deletion remain explicitly out of scope.
