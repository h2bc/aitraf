# Feature Specification: Publish Demo Assets

**Feature Branch**: `[007-publish-demo-assets]`

**Created**: 2026-07-11

**Status**: Draft

**Input**: User description: "On API startup, publish the selected demo videos
and thumbnails to a public bucket when they do not already exist, then serve
constant public links for both asset types instead of expiring links."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Serve Stable Public Demo Assets (Priority: P1)

As a demo user, I need each prediction to reference stable public video and
thumbnail links so media remains directly accessible without expiring links.

**Why this priority**: Stable media access is the primary user-visible reason
for publishing the selected demo assets.

**Independent Test**: Start the API with a valid prediction subset and public
asset configuration, request demo predictions, and verify every item contains
working, non-expiring public video and thumbnail links.

**Acceptance Scenarios**:

1. **Given** all selected videos and thumbnails already exist publicly, **When**
   the API starts, **Then** it becomes ready without replacing those objects and
   returns their stable public links.
2. **Given** a selected video is absent from public storage, **When** the API
   starts, **Then** that video is published from its configured source before
   the API becomes ready.
3. **Given** a selected thumbnail is absent from public storage, **When** the API
   starts, **Then** the source video is obtained, the thumbnail is created and
   published, and its stable public link is returned.

---

### User Story 2 - Publish Only The Selected Subset (Priority: P2)

As a demo operator, I need startup to publish only videos referenced by the
active prediction subset so unrelated training and evaluation media is not made
public.

**Why this priority**: The public surface must remain intentionally limited to
the curated demo rather than expose the wider dataset.

**Independent Test**: Supply prediction rows referencing a known subset and
verify that only those videos and their thumbnails are checked or published.

**Acceptance Scenarios**:

1. **Given** source storage contains selected and unselected videos, **When** the
   API starts, **Then** only assets referenced by matched demo prediction rows
   are eligible for publication.
2. **Given** classification and assessment results do not match for a video,
   **When** the active subset is formed, **Then** that video's assets are not
   published merely because one source references it.

---

### User Story 3 - Restart Safely And Predictably (Priority: P3)

As an operator, I need repeated starts and model redeployments to preserve
already-published assets and fail clearly when required publication cannot be
completed.

**Why this priority**: Startup publishing must not damage working production
media or allow the API to report readiness with broken links.

**Independent Test**: Start the API twice with the same subset, verify the
second start performs no asset writes, and verify a publication failure blocks
readiness with an actionable error.

**Acceptance Scenarios**:

1. **Given** the same valid subset is used across multiple starts, **When** the
   API starts again, **Then** existing public objects are reused without being
   overwritten.
2. **Given** a required source is missing, invalid, or cannot be published,
   **When** the API starts, **Then** readiness fails explicitly and no response
   containing a known-broken asset link is served.
3. **Given** a later deployment selects additional videos, **When** it starts,
   **Then** only missing assets are added and assets from the previous subset
   remain unchanged.

### Edge Cases

- Public asset bucket configuration is missing or inconsistent with the shared
  storage endpoint.
- A prediction row has a missing, malformed, or cross-source video identifier.
- The source video does not exist or cannot be read.
- The public video exists but its thumbnail does not, or vice versa.
- Thumbnail creation produces no valid image.
- Two rows refer to the same asset and must not trigger duplicate publication.
- An object appears between the existence check and publication attempt.
- A public object already exists at the expected identity but does not represent
  the configured source; the system reports a conflict rather than overwriting it.
- Public storage is reachable for reads but rejects publication.
- Previously published assets are no longer selected; startup leaves them
  untouched because retention and deletion are outside this feature.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST derive the publishable demo asset set exclusively
  from the matched prediction rows selected for serving.
- **FR-002**: Before accepting demo traffic, the system MUST ensure that every
  selected video exists in the configured public asset location.
- **FR-003**: When a selected public video is missing, the system MUST publish it
  from the source location recorded by its prediction row.
- **FR-004**: Before accepting demo traffic, the system MUST ensure that every
  selected video has a corresponding public thumbnail.
- **FR-005**: When a selected public thumbnail is missing, the system MUST obtain
  the source video, generate the thumbnail, and publish it.
- **FR-006**: Existing public videos and thumbnails MUST be reused and MUST NOT
  be overwritten during normal startup publication.
- **FR-007**: Publication MUST be idempotent: repeating startup with identical
  inputs MUST produce the same asset identities and public links without
  additional writes.
- **FR-008**: Demo prediction responses MUST expose stable public links for both
  video and thumbnail assets and MUST NOT generate expiring access links.
- **FR-009**: Public asset identities MUST be deterministic from the selected
  asset identity and asset type so existence can be checked before publication.
- **FR-010**: The system MUST reject missing or invalid configuration, malformed
  asset records, unavailable source assets, publication failures, thumbnail
  failures, and existing-object conflicts with explicit errors.
- **FR-011**: The API MUST NOT become ready when any selected asset lacks a valid
  public object and stable public link.
- **FR-012**: Startup MUST NOT publish videos that are absent from the matched
  prediction subset.
- **FR-013**: Startup MUST NOT delete public assets that are absent from the
  current subset.
- **FR-014**: Retraining, selecting production prediction runs, asset retention,
  and garbage collection MUST remain outside this feature.
- **FR-015**: The old response-time expiring-link behavior and its serving-only
  configuration MUST be removed rather than retained as an alternate path.

### Architecture And Reuse Requirements *(mandatory)*

- **AR-001**: Startup publication and response composition MUST remain owned by
  `packages/aitraf-api`; shared dependency-light storage primitives used by more
  than one package MAY remain in `packages/aitraf-core`.
- **AR-002**: The feature MUST extend the existing demo prediction startup,
  video, thumbnail, and service surfaces rather than introduce a parallel
  publisher or training workflow.
- **AR-003**: Public identity creation, URL construction, existence checks, and
  publication decisions MUST use reusable single-purpose helpers rather than
  being duplicated between video and thumbnail flows.
- **AR-004**: Production publication behavior MUST live in versioned repository
  code and MUST NOT depend on a one-time manual or notebook-only command.
- **AR-005**: Asset-set and public-link transformations MUST use explicit inputs
  and outputs, with storage and startup state localized at their boundaries.
- **AR-006**: Callers, tests, documentation, and validation commands MUST move
  directly to stable public links; obsolete presigning paths, parameters, and
  dead code MUST be removed.
- **AR-007**: Publication MUST accept one required prediction-row asset schema
  and one required public-asset configuration schema, rejecting alternate
  shapes rather than normalizing them.

### Validation And Reproducibility Requirements *(mandatory)*

- **VR-001**: Automated tests MUST cover existing-object reuse, missing-video
  publication, missing-thumbnail generation, subset-only publication,
  deterministic links, idempotent restart, and explicit startup failures.
- **VR-002**: Command-level smoke validation MUST start the API against a known
  prediction subset and verify that every returned public video and thumbnail
  link is reachable without signed query parameters.
- **VR-003**: Verification MUST record the prediction run identifiers, matched
  rows, source and public storage configuration, public base URL, and commands
  required to reproduce publication.
- **VR-004**: Validation MUST prove that the second startup with identical inputs
  performs zero public asset writes and that unselected source videos are not
  published.
- **VR-005**: Validation MUST prove at least one missing-source, publication,
  existing-object conflict, and thumbnail-generation failure prevents readiness
  with explicit context.

### Key Entities *(include if feature involves data)*

- **Selected Demo Asset**: A video chosen through matched demo prediction rows;
  includes its video identifier, source location, and derived public identities.
- **Public Video**: The publicly readable copy of one selected source video;
  includes a deterministic identity and stable public link.
- **Public Thumbnail**: The publicly readable thumbnail corresponding to one
  selected video; includes a deterministic identity and stable public link.
- **Public Asset Configuration**: The required runtime settings identifying the
  public bucket on the shared storage endpoint.
- **Prepared Prediction Row**: A served prediction row enriched with required
  stable public video and thumbnail links after startup publication succeeds.

## Architecture And Data Impact

- **Touched Surfaces**: `packages/aitraf-api` application startup, demo prediction
  video and thumbnail preparation, response composition, runtime configuration,
  tests, deployment examples, and documentation; dependency-light shared storage
  helpers in `packages/aitraf-core` only if genuinely reused.
- **Shared Helpers To Add Or Extend**: Deterministic public asset identity and URL
  construction, strict public configuration loading, source-to-public object
  publication, object existence checks, and publication conflict reporting.
- **Legacy Surfaces Removed**: Response-time video and thumbnail presigning,
  expiring-link configuration, and any private-asset response fields superseded
  by stable public links.
- **Required Input Shapes**: Each matched prediction row must contain exactly the
  required video identity and source location fields; public configuration must
  contain one public bucket identity alongside the existing shared endpoint.
  Missing, extra
  alternate, or malformed representations fail at their owning boundary.
- **Data Or Artifact Impact**: The selected source videos are copied to a public
  asset location and missing thumbnails are generated there. Prediction and
  model artifacts are unchanged. Existing public objects are preserved and
  unselected public objects are not deleted.
- **Reproducibility Inputs**: Prediction run identifiers and fixed prediction
  artifact references, matched prediction rows, source and public asset
  configuration, startup command, and smoke-validation command.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of returned demo predictions contain reachable, stable public
  links for both video and thumbnail assets before readiness succeeds.
- **SC-002**: Repeating startup with identical inputs causes zero video or
  thumbnail writes and returns exactly the same public links.
- **SC-003**: 100% of source videos outside the matched demo subset remain
  unpublished by this workflow.
- **SC-004**: 100% of covered missing configuration, missing source, publication,
  object-conflict, and thumbnail-generation failures prevent readiness and
  identify the affected asset.
- **SC-005**: Demo responses contain zero expiring media links or signed media
  query parameters after migration.
- **SC-006**: Adding one new selected video publishes at most that missing video
  and its missing thumbnail while leaving all previously published objects
  unchanged.

## Assumptions

- Anonymous public access to the destination assets is intentional and approved.
- The public destination uses the existing storage endpoint and credentials;
  only `AITRAF_PUBLIC_ASSET_BUCKET` is added as required configuration.
- The matched prediction subset is the authoritative publication allowlist.
- Source and public destinations are independently configured; source objects
  remain private and unchanged.
- Existing public objects are immutable under normal startup. A conflicting
  object is an error, not permission to overwrite it.
- Thumbnail creation requires temporary local access to a source video; videos
  need not otherwise be downloaded when they can be copied by the storage
  service.
- Retention and removal of public assets are handled by a future explicit
  lifecycle policy, not implicitly during startup.
