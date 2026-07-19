# Feature Specification: Pose-Rendered Demo Assets

**Feature Branch**: `[009-pose-rendered-assets]`

**Created**: 2026-07-18

**Status**: Draft

**Input**: User description: "In the API, instead of returning video and
thumbnail links without poses, return the same thing with poses."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - See The Model's Pose Evidence (Priority: P1)

As a demo viewer, I want the clip and preview image for each prediction to show
the detected skeleton drawn over the rider, so I can see the body positions the
model actually observed instead of raw footage.

**Why this priority**: This is the whole feature.

**Independent Test**: Request the demo predictions listing, open the video and
thumbnail links for any prediction, and verify both show the skeleton overlay.

**Acceptance Scenarios**:

1. **Given** the demo predictions listing, **When** it is requested, **Then**
   each prediction's video link and thumbnail link resolve to pose-rendered
   assets.
2. **Given** a pose-rendered video link, **When** it is opened without
   credentials, **Then** the clip plays with the skeleton overlay tracking the
   rider.
3. **Given** a pose-rendered thumbnail link, **When** it is opened, **Then** it
   is a still image showing the overlay.
4. **Given** the listing response, **When** its shape is inspected, **Then** it
   has the same fields as before; only what the video and thumbnail links point
   at has changed.

### Edge Cases

- Pose data is missing, unreadable, or empty for a selected video; startup fails
  explicitly rather than serving the plain video as a substitute.
- Pose data does not line up with its source video (frame count, duration, or
  orientation); the mismatch fails explicitly rather than producing a misaligned
  overlay.
- A frame contains no detected person; that frame renders without an overlay and
  the clip continues. This is not a failure.
- A frame contains multiple detected people; one defined rule decides what is
  drawn, applied consistently across every clip.
- A source video is rotated by container metadata; the overlay follows the
  displayed orientation, not the stored frame orientation.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The video and thumbnail links returned for each demo prediction
  MUST point at pose-rendered assets showing the skeleton overlay drawn over the
  source footage.
- **FR-002**: The response MUST keep its existing shape and field names; only the
  content the links resolve to changes.
- **FR-003**: A pose-rendered video MUST match the source footage in duration and
  displayed orientation, differing only by the overlay. A pose-rendered thumbnail
  MUST be taken from that footage at the same source moment used today.
- **FR-004**: The system MUST resolve each prediction's pose data through a
  deterministic association derived from the prediction's video identity, never
  by ordering, positional matching, or fuzzy name matching.
- **FR-005**: Missing, unreadable, empty, or inconsistent pose data for a
  selected prediction MUST fail explicitly at startup. The system MUST NOT report
  readiness, omit the prediction, or fall back to the non-pose asset.
- **FR-006**: Pose-rendered assets MUST be published once at startup under
  deterministic keys and served as stable non-expiring links requiring no
  credentials, exactly as the current assets are. The listing operation MUST
  perform no rendering, existence checking, uploading, or signing while serving a
  request.
- **FR-007**: Preparation MUST be idempotent: an existing valid pose-rendered
  asset is reused with no re-render and no re-upload.
- **FR-008**: The non-pose video and thumbnail assets MUST no longer be produced
  or served by this feature. Serving both variants, viewer-selectable overlays,
  and request-time pose extraction are out of scope.

### Architecture And Reuse Requirements *(mandatory)*

- **AR-001**: The change MUST be owned by the existing demo predictions feature
  surface in `packages/aitraf-api`, extending the current startup asset-publishing
  flow rather than adding a parallel pipeline. It MUST NOT make the serving
  package depend on the offline training package.
- **AR-002**: Pose drawing MUST exist as exactly one helper, owned by the feature
  surface that consumes it, replacing the training-only and notebook-only copies
  rather than duplicating them. It MUST NOT be promoted to a shared package
  unless more than one production surface needs it.
- **AR-003**: Publishing MUST reuse the existing single-purpose ensure-and-publish
  helpers rather than adding a duplicated copy-and-upload branch.
- **AR-004**: The feature MUST define one required pose data shape and one
  required prediction-to-pose association, rejecting alternate shapes instead of
  normalizing them.
- **AR-005**: The superseded non-pose asset path MUST be removed, not retained
  behind a flag, alias, or alternate field.

### Validation And Reproducibility Requirements *(mandatory)*

- **VR-001**: Automated tests MUST cover the prediction-to-pose association,
  first-time preparation, reuse without re-render, missing pose data,
  inconsistent pose data, and frames without detections.
- **VR-002**: Command-level smoke validation MUST start against an empty public
  asset store, verify assets are produced and published, restart, and verify no
  re-render occurs and previously issued links still resolve.
- **VR-003**: Visual validation MUST confirm on each selected clip that the
  overlay tracks the rider and stays aligned from first to last frame.
- **VR-004**: Failure validation MUST prove that missing and inconsistent pose
  data each block readiness with an explicit error rather than serving a non-pose
  asset.
- **VR-005**: Verification MUST record the prediction source run identities, the
  pose data location, the public asset store configuration, and the commands
  required to reproduce the result.

### Key Entities *(include if feature involves data)*

- **Demo Prediction**: One selected video with its ground truth, model
  predictions, and its published video and thumbnail links.
- **Pose Data**: Per-frame detected keypoints for one source video, produced by
  the offline extraction pipeline and associated to a prediction by video
  identity.

## Architecture And Data Impact

- **Touched Surfaces**: The demo predictions feature in `packages/aitraf-api`
  (startup asset preparation, thumbnail generation, configuration), plus a shared
  runtime package for the pose drawing helper, tests, contracts, and API docs.
  The response schema itself is unchanged. The separately deployed frontend is an
  external consumer and needs no change.
- **Shared Helpers To Add Or Extend**: A pose overlay drawing helper promoted out
  of training-only and notebook-only code; a clip renderer that writes overlaid
  footage; the existing ensure-and-publish helpers extended to cover the rendered
  assets.
- **Legacy Surfaces Removed**: The non-pose video and thumbnail publishing path;
  the training-only and notebook-only copies of the pose drawing helper.
- **Required Input Shapes**: A prediction row carries a video identity that
  deterministically resolves both its source video and its pose data. Pose data
  is one required per-frame keypoint representation. Missing, empty, alternate,
  or inconsistent shapes fail explicitly.
- **Data Or Artifact Impact**: Introduces a read dependency on the extracted pose
  artifacts and changes what the published demo assets contain. Model, training,
  evaluation, and tracking artifacts are unchanged.
- **Reproducibility Inputs**: Pinned prediction source run identities, pose
  artifact location, public asset store configuration, key layout, overlay
  selection and styling rules, and application start and restart commands.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of returned predictions carry video and thumbnail links that
  resolve without credentials and show the overlay.
- **SC-002**: 100% of rendered clips match their source footage in duration and
  displayed orientation, with the overlay tracking the rider across the full clip.
- **SC-003**: A restart against a fully prepared asset store performs zero renders
  and zero uploads, and previously issued links still resolve.
- **SC-004**: 100% of tested missing and inconsistent pose inputs block readiness
  with an explicit error and never yield a listing containing a non-pose asset.
- **SC-005**: Listing response time is unchanged by this feature.

## Assumptions

- The pose-rendered assets replace the non-pose assets. Consumers are not offered
  both.
- Every video selected for the demo listing already has extracted pose data.
  Producing pose data for videos that lack it is out of scope; encountering such a
  video is an explicit failure.
- The prediction's existing video identity is enough to deterministically resolve
  its pose data. If the pose artifact location is not derivable from that identity
  alone, supplying it is a configuration input.
- Where rendering runs, how pose data reaches the serving environment, whether the
  published keys are reused or renamed, and which package owns the shared helper
  are all planning decisions. This spec constrains only the outcome and the
  failure behavior.
- Overlay appearance uses one fixed repository-defined style.
- Invalid or missing pose state is surfaced as an explicit startup failure rather
  than repaired or masked with a fallback asset.
