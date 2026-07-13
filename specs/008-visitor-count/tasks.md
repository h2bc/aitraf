# Tasks: Visitor Count

**Input**: Design documents from `/specs/008-visitor-count/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/visitor-count.md, quickstart.md

**Validation**: Automated tests and command-level smoke validation are required by the feature specification. Production deployment configuration is owned by another repository and is explicitly outside this task list.

**Organization**: Tasks are grouped by user story so the page-view endpoint, persistence guarantees, and public browser contract can be implemented and validated incrementally.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it changes different files and has no dependency on another incomplete task in the same phase
- **[Story]**: Maps a task to User Story 1, 2, or 3
- Every task includes an exact repository path

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add the Redis dependency and reproducible local service without changing the external production deployment repository

- [X] T001 Add redis-py to `packages/aitraf-api/pyproject.toml` and refresh the resolved dependency state in `uv.lock`
- [X] T002 [P] Add the local API and Redis 7 Compose services with source reload, AOF `everysec`, periodic snapshots, health checks, and a named Redis volume in `compose.yaml`
- [X] T003 Register the `redis_integration` pytest marker in `packages/aitraf-api/pyproject.toml`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish strict Redis configuration and the API-owned counter boundary required by every story

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Add required `redis_url: str` settings loading from `AITRAF_REDIS_URL` in `packages/aitraf-api/src/aitraf_api/config.py`
- [X] T005 [P] Add missing, empty, and valid `AITRAF_REDIS_URL` settings coverage in `packages/aitraf-api/tests/test_config.py`
- [X] T006 [P] Add the devcontainer-local `AITRAF_REDIS_URL` to `.env.example`
- [X] T007 Create the `packages/aitraf-api/src/aitraf_api/features/visitor_count/__init__.py` feature package and exports
- [X] T008 Define the narrow async counter protocol, fixed `aitraf:visitor-count:homepage` identity, Redis error types, and strict non-negative increment-result validation in `packages/aitraf-api/src/aitraf_api/features/visitor_count/service.py`
- [X] T009 [P] Add reusable fake counter and visitor-count app fixtures in `packages/aitraf-api/tests/features/visitor_count/conftest.py`
- [X] T010 Update existing `Settings` constructors and app fixtures for the required Redis URL in `packages/aitraf-api/tests/test_app.py` and `packages/aitraf-api/tests/features/demo_predictions/conftest.py`

**Checkpoint**: Required configuration and a single typed API-owned counter boundary are ready; no in-memory production fallback or shared-core abstraction exists

---

## Phase 3: User Story 1 - Record And Display A Visit (Priority: P1) 🎯 MVP

**Goal**: Record one page view with one atomic increment and return the resulting lifetime count

**Independent Test**: Start from a fake count of 41, submit one valid request, verify `200 {"count": 42}`, then submit another request and verify the result is 43

### Validation for User Story 1 ⚠️

- [X] T011 [P] [US1] Add service tests for missing-key first increment, sequential increments, exact result mapping, and invalid negative/out-of-range results in `packages/aitraf-api/tests/features/visitor_count/test_service.py`
- [X] T012 [P] [US1] Add route contract tests for `POST /visitor-count`, strict `{"count": integer}` output, and one increment per request in `packages/aitraf-api/tests/features/visitor_count/test_route.py`

### Implementation for User Story 1

- [X] T013 [P] [US1] Define the strict visitor-count response schema in `packages/aitraf-api/src/aitraf_api/features/visitor_count/schemas.py`
- [X] T014 [US1] Implement Redis-backed atomic `INCR` and explicit Redis/result failures in `packages/aitraf-api/src/aitraf_api/features/visitor_count/service.py`
- [X] T015 [US1] Implement `POST /visitor-count` with no bearer dependency and response-model enforcement in `packages/aitraf-api/src/aitraf_api/features/visitor_count/route.py`
- [X] T016 [US1] Register the visitor-count router alongside existing health and demo routes in `packages/aitraf-api/src/aitraf_api/features/__init__.py`

**Checkpoint**: User Story 1 is independently usable with an injected counter and returns the increment result without a separate read

---

## Phase 4: User Story 2 - Preserve An Accurate Total (Priority: P2)

**Goal**: Use one shared Redis counter safely across concurrent requests and preserve it across normal API and Redis container restarts

**Independent Test**: Against local Redis, run 100 concurrent increments and observe exactly 100 additional counts, restart Redis without deleting its volume, and verify the next increment continues from the prior value

### Validation for User Story 2 ⚠️

- [X] T017 [P] [US2] Add real-Redis integration tests for 100 concurrent `INCR` operations, missing-key creation, invalid stored types/values, and unavailable Redis in `packages/aitraf-api/tests/features/visitor_count/test_redis_integration.py`
- [X] T018 [P] [US2] Add application tests for Redis startup validation, invalid stored state blocking readiness, shared client injection, and lifecycle closure in `packages/aitraf-api/tests/test_app.py`

### Implementation for User Story 2

- [X] T019 [US2] Add Redis client construction, ping and existing-key validation, application-state injection, and shutdown cleanup through FastAPI lifespan in `packages/aitraf-api/src/aitraf_api/app.py`
- [X] T020 [US2] Map request-time Redis connection/operation failures to explicit `503` responses and invalid stored/result state to explicit `500` responses in `packages/aitraf-api/src/aitraf_api/features/visitor_count/route.py`
- [X] T021 [US2] Add local Compose start, health, integration-test, concurrent-increment, and named-volume restart commands with expected results to `specs/008-visitor-count/quickstart.md`

**Checkpoint**: User Stories 1 and 2 work with real Redis without lost concurrent updates or process-local state

---

## Phase 5: User Story 3 - Support A Public Web Page Safely (Priority: P3)

**Goal**: Allow the frontend to record a page load without a private token or visitor identity while rejecting unsupported request shapes

**Independent Test**: Call the endpoint without authorization or visitor data and receive the new count; send a body or query parameter and receive `422` without an increment; verify `/demo-predictions` still returns `401` without its bearer token

### Validation for User Story 3 ⚠️

- [X] T022 [P] [US3] Add public-access, no-credential, no-visitor-data, body rejection, query rejection, and unchanged protected-route authentication tests in `packages/aitraf-api/tests/features/visitor_count/test_route.py`

### Implementation for User Story 3

- [X] T023 [US3] Enforce the empty request contract and reject all request bodies and query parameters without incrementing in `packages/aitraf-api/src/aitraf_api/features/visitor_count/route.py`
- [X] T024 [US3] Document the public browser-call contract, retry/page-view semantics, privacy boundary, and external `AITRAF_REDIS_URL` deployment handoff in `packages/aitraf-api/README.md`

**Checkpoint**: All three stories are functional; the endpoint is public and strict while existing API authentication remains unchanged

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Finish repository-only documentation and prove the complete feature without modifying production deployment files

- [X] T025 [P] Update root local-development Redis startup and teardown guidance in `README.md`
- [X] T026 Validate concurrency with the isolated real-Redis integration coverage in `packages/aitraf-api/tests/features/visitor_count/test_redis_integration.py`
- [X] T027 Run `docker compose config --quiet`, the full API pytest suite, Redis-marked integration tests, lint checks, and a single-request endpoint smoke check; record commands and results in `specs/008-visitor-count/validation.md`
- [X] T028 Verify with `git diff --name-only` and document in `specs/008-visitor-count/validation.md` that no external or production deployment Compose, proxy, network, or volume configuration was added to this repository
- [X] T029 Audit `packages/aitraf-api/src/aitraf_api/features/visitor_count/` for in-memory fallbacks, alternate keys/endpoints, visitor identity collection, broad input normalization, and dead compatibility paths; record the clean result in `specs/008-visitor-count/validation.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Starts immediately; T002 can proceed while T001 and T003 update package metadata sequentially
- **Foundational (Phase 2)**: Depends on Phase 1 and blocks all user stories
- **User Story 1 (Phase 3)**: Depends on Phase 2 and delivers the MVP contract
- **User Story 2 (Phase 4)**: Depends on the US1 counter implementation and endpoint, then adds real Redis lifecycle and concurrency behavior
- **User Story 3 (Phase 5)**: Depends on the US1 endpoint, but its contract tests and documentation can proceed in parallel with US2
- **Polish (Phase 6)**: Depends on all selected user stories

### User Story Dependencies

```text
Setup → Foundation → US1 (MVP) → US2
                         └──────→ US3
US2 + US3 → Polish
```

- **US1**: First deliverable after foundation; no dependency on later stories
- **US2**: Reuses US1's atomic counter and endpoint but is independently verified against real Redis
- **US3**: Reuses US1's endpoint but can be completed independently of US2 after US1 is ready

### Within Each User Story

- Write the listed validation coverage before or alongside implementation
- Implement schemas/protocols before route wiring
- Keep Redis state and lifecycle at the application/service boundary
- Complete each checkpoint before beginning a dependent phase

### Parallel Opportunities

- T002 can run in parallel with the package metadata work in T001 and T003
- T005, T006, and T009 touch independent configuration/test files
- T011 and T012 define US1 service and route behavior independently
- T017 and T018 split real-Redis behavior from application lifecycle behavior
- After US1, US2 and US3 can proceed in parallel
- T025 can proceed independently while T026 is created

---

## Parallel Example: User Story 1

```text
Task T011: Define service-level increment and validation behavior in test_service.py
Task T012: Define endpoint response and one-call/one-increment behavior in test_route.py
Task T013: Define the response schema in schemas.py
```

## Parallel Example: User Stories 2 And 3

```text
US2: Implement and test real Redis concurrency, startup validation, and lifecycle cleanup
US3: Test and enforce the public empty-request contract and update API documentation
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Setup and Foundational phases.
2. Complete User Story 1 service, schema, route, and tests.
3. Validate one request increments and returns the new count.
4. Stop for MVP review before adding lifecycle hardening and the public strictness checks.

### Incremental Delivery

1. **Foundation**: Required Redis config, local service, and typed counter boundary.
2. **US1**: Atomic increment endpoint and returned count.
3. **US2**: Real Redis lifecycle, concurrent safety, and restart persistence.
4. **US3**: Public browser contract, privacy boundary, and strict empty input.
5. **Polish**: Documentation, smoke validation, scope audit, and recorded evidence.

### Repository Boundary

- Implement local `compose.yaml` only.
- Do not create or edit production deployment Compose, proxy, network, or volume configuration in this repository.
- Expose `AITRAF_REDIS_URL` as the sole production deployment handoff.

## Notes

- `[P]` tasks operate on independent files or test surfaces.
- Every accepted request is a page view; retries may increment again.
- Redis `INCR` prevents lost concurrent updates but does not provide request idempotency.
- AOF `everysec` and the named local volume prove normal-restart persistence, not off-host backup or zero-loss crash durability.
- Keep all production counter code in `aitraf-api`; do not add a generic Redis abstraction to `aitraf-core`.
