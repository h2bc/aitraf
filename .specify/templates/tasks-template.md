---

description: "Task list template for feature implementation"
---

# Tasks: [FEATURE NAME]

**Input**: Design documents from `/specs/[###-feature-name]/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Validation**: Validation tasks are REQUIRED. Add automated tests when practical, and always add command-level smoke validation for pipeline behavior.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **API package**: `packages/aitraf-api/src/aitraf_api/`,
  `packages/aitraf-api/tests/`
- **Core runtime package**: `packages/aitraf-core/src/aitraf_core/`
- **Train package configs**: `packages/aitraf-train/configs/`
- **Train package entrypoints**: `packages/aitraf-train/scripts/`
- **Train package code**: `packages/aitraf-train/src/aitraf_train/`
- **Validation**: package-local `tests/`, `tests/unit/`,
  `tests/integration/`, `tests/smoke/`
- **Analysis only**: `notebooks/` (do not leave production-only behavior here)

<!--
  ============================================================================
  IMPORTANT: The tasks below are SAMPLE TASKS for illustration purposes only.

  The /speckit-tasks command MUST replace these with actual tasks based on:
  - User stories from spec.md (with their priorities P1, P2, P3...)
  - Feature requirements from plan.md
  - Entities from data-model.md
  - Endpoints from contracts/

  Tasks MUST be organized by user story so each story can be:
  - Implemented independently
  - Tested independently
  - Delivered as an MVP increment

  DO NOT keep these sample tasks in the generated tasks.md file.
  ============================================================================
-->

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and constitution-aligned scaffolding

- [ ] T001 Create project structure per implementation plan
- [ ] T002 Update the relevant config surface in `configs/`
- [ ] T003 [P] Add or update shared validation/linting scaffolding if needed

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

Examples of foundational tasks (adjust based on your project):

- [ ] T004 Extend helpers in the owning package, or shared package only when multiple feature surfaces need them
- [ ] T005 [P] Add or update config wiring and task/model dispatch paths
- [ ] T006 [P] Add explicit error handling for invalid config, data, or unsupported states
- [ ] T007 Define reusable data/model/metric interfaces as functional helpers with explicit inputs and outputs
- [ ] T008 Configure logging, tracking, or artifact outputs required across stories
- [ ] T009 Document reproducibility inputs and command surfaces

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - [Title] (Priority: P1) 🎯 MVP

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Validation for User Story 1 ⚠️

> **NOTE**: Define validation before implementation. Add automated tests when practical and always add at least one smoke validation command.

- [ ] T010 [P] [US1] Add unit/integration coverage in `tests/` for the changed helper or pipeline behavior
- [ ] T011 [P] [US1] Add a smoke validation command and expected outcome for the user journey

### Implementation for User Story 1

- [ ] T012 [P] [US1] Add or extend decomposed helper functions in the owning package/feature module
- [ ] T013 [P] [US1] Update the relevant package/task/model/config wiring without introducing parallel architecture
- [ ] T014 [US1] Implement the user-facing behavior in the appropriate package route/service/script/task surface
- [ ] T015 [US1] Add explicit failure handling for invalid or unsupported states
- [ ] T016 [US1] Add tracking/logging/artifact output changes required for verification
- [ ] T017 [US1] Update docs/spec references for the new command/config behavior

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - [Title] (Priority: P2)

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Validation for User Story 2 ⚠️

- [ ] T018 [P] [US2] Add unit/integration coverage in `tests/` for the changed helper or pipeline behavior
- [ ] T019 [P] [US2] Add a smoke validation command and expected outcome for the user journey

### Implementation for User Story 2

- [ ] T020 [P] [US2] Add or extend decomposed helper functions in the owning package/feature module
- [ ] T021 [US2] Update the relevant task/model/config wiring without duplicating existing logic
- [ ] T022 [US2] Implement the user-facing pipeline behavior in the appropriate repo surface
- [ ] T023 [US2] Integrate through shared components instead of story-specific parallel paths

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - [Title] (Priority: P3)

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Validation for User Story 3 ⚠️

- [ ] T024 [P] [US3] Add unit/integration coverage in `tests/` for the changed helper or pipeline behavior
- [ ] T025 [P] [US3] Add a smoke validation command and expected outcome for the user journey

### Implementation for User Story 3

- [ ] T026 [P] [US3] Add or extend decomposed helper functions in the owning package/feature module
- [ ] T027 [US3] Update the relevant task/model/config wiring without duplicating existing logic
- [ ] T028 [US3] Implement the user-facing pipeline behavior in the appropriate repo surface

**Checkpoint**: All user stories should now be independently functional

---

[Add more user story phases as needed, following the same pattern]

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] TXXX [P] Documentation updates in docs/
- [ ] TXXX Code cleanup and refactoring
- [ ] TXXX Performance optimization across all stories
- [ ] TXXX [P] Additional validation coverage in `tests/unit/`, `tests/integration/`, or `tests/smoke/`
- [ ] TXXX Remove duplicated logic or architecture drift discovered during implementation
- [ ] TXXX Run documented smoke commands and verify expected artifacts/logs

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Validation tasks MUST be defined before implementation
- Shared helpers before task/model-specific wiring, but only promote shared code
  when more than one feature surface needs it
- Config and architecture updates before command-surface integration
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All validation tasks for a user story marked [P] can run in parallel
- Decomposed helper tasks within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch validation tasks for User Story 1 together:
Task: "Add unit/integration coverage in tests/"
Task: "Add a smoke validation command and expected outcome for the user journey"

# Launch decomposed helper work for User Story 1 together:
Task: "Add or extend helper function in packages/<owning-package>/src/<module_a>/"
Task: "Add or extend helper function in packages/<owning-package>/src/<module_b>/"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Prefer package-by-feature ownership over adding new parallel structure
- Prefer functional programming: small, pure helpers with explicit inputs and outputs
- Ensure invalid states fail explicitly instead of relying on fallback behavior
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
