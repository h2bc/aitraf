---

description: "Task list for pose-rendered demo assets"
---

# Tasks: Pose-Rendered Demo Assets

**Input**: Design documents from `/specs/009-pose-rendered-assets/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/)

**Validation**: Required. Automated tests plus command-level smoke validation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to
- 🧑 **CHECKPOINT**: A run-and-verify step for the user, not a code edit. Do not
  proceed past it until its expected outcome is observed.

## Stage Mapping

| Stage | Phases | Deliverable |
|-------|--------|-------------|
| **Stage 1** — offline pose upload | Phase 1–2 | `.npz` artifacts in the private bucket, verified idempotent |
| **Stage 2** — API rendering | Phase 3–5 | Pose-rendered assets served, verified end to end |

Stage 2 cannot start until 🧑 **T010** confirms poses exist in the bucket.

---

## Phase 1: Setup

**Purpose**: Scaffolding needed by Stage 1.

- [X] T001 Create `packages/aitraf-train/src/aitraf_train/storage/artifacts.py` with a module docstring and empty `__all__`
- [X] T002 [P] Create `packages/aitraf-train/tests/storage/` with `__init__.py`, mirroring the existing test tree layout

---

## Phase 2: Foundational — Offline Pose Upload (Stage 1)

**Purpose**: Get pose artifacts into the private bucket. Blocks all API work —
the API has nothing to download until this is done and run.

**Independent test**: Run the pipeline with only `upload_poses` enabled; confirm
`.npz` objects appear under `poses/`; re-run and confirm every file is skipped.

### Shared helper

- [X] T003 Implement `upload_directory(directory, *, bucket, prefix, force, s3_client) -> int` in `packages/aitraf-train/src/aitraf_train/storage/artifacts.py` — walk files in sorted order, derive `{prefix}/{relative_path}`, skip when `object_exists` and not `force`, upload via `upload_s3_uri`, log progress every 10%, log a summary, and **raise on the first upload failure** (research R-008; do not replicate `upload_pairs`' warn-and-count)
- [X] T004 Raise explicitly in `upload_directory` when the directory is missing, is not a directory, contains no files, or `prefix` is empty — matching `upload_pairs.py:37-51`
- [X] T005 [P] Unit-test `upload_directory` in `packages/aitraf-train/tests/storage/test_artifacts.py` with a stubbed S3 client: fresh upload, skip-existing, `force=true` overwrite, raise-on-failure, and each input-validation failure

### Refactor the existing caller

- [X] T006 Rewrite `upload_pairs` in `packages/aitraf-train/src/aitraf_train/preparation/label_ops/upload_pairs.py` to delegate to `upload_directory`, removing its duplicated walk/skip/progress/summary loop. **Behavior change**: per-file failures now raise instead of being counted (research R-008). Update any existing pairs-upload test to the new behavior.

### New step

- [X] T007 Create `packages/aitraf-train/src/aitraf_train/preparation/data_ops/upload_poses.py` with `PoseUploadConfig(poses_dir, prefix, force)` — paths coerced and `prefix` stripped in `__post_init__`, mirroring `PairUploadConfig` — and `upload_poses(config)` calling `load_s3_settings(require_bucket=True)` then `upload_directory`. Poses only; boxes stay local (research R-010).
- [X] T008 Add the `upload_poses` block to `packages/aitraf-train/configs/data_ops.yaml` with `enabled: true`, `force: false`, `poses_dir: ${paths.poses_dir}`, `prefix: poses`
- [X] T009 Add the `upload_poses` clause to `packages/aitraf-train/scripts/data_ops_pipeline.py` — import, `if data_cfg.upload_poses.enabled:` + `heading("Upload Poses")`, else-branch skip message — positioned **after** `pose_and_bbox_extraction`

### 🧑 Stage 1 checkpoint

- [X] T010 🧑 **CHECKPOINT — run it for real.** Confirm `storage/data/poses/*.npz` exists locally, then run the upload with all other steps disabled (command in [quickstart.md](./quickstart.md) §1). **Verify**: first run logs `N uploaded, 0 skipped`; second run logs `0 uploaded, N skipped`; `aws s3 ls s3://$AWS_BUCKET/poses/` lists the files. If poses are not available locally, run `pose_and_bbox_extraction` first — Stage 2 cannot proceed without this.

**Stage 1 complete.** Poses are in the bucket.

---

## Phase 3: User Story 1 — Pose-Rendered Assets (P1) — Stage 2

**Goal**: The video and thumbnail links resolve to assets showing the skeleton
overlay.

**Independent test**: Request the listing, open both links for any prediction,
verify the overlay is visible and the response shape is unchanged.

### Render helpers in `aitraf-api`

- [X] T011 [US1] Add `numpy`, `pillow`, `imageio`, and `av` to `packages/aitraf-api/pyproject.toml` dependencies and refresh `uv.lock`. Do **not** add `torch`, `ultralytics`, or `aitraf-ml-core` (plan Constraints).
- [X] T012 [US1] Move `packages/aitraf-train/src/aitraf_train/utils/draw_utils.py` to `packages/aitraf-api/src/aitraf_api/features/demo_predictions/render/draw.py` verbatim; create `packages/aitraf-api/src/aitraf_api/features/demo_predictions/render/__init__.py` exporting `POSE_DEFAULT_SKELETON`, `draw_pose_keypoints`, `draw_bounding_boxes`
- [X] T013 [US1] Delete the original `draw_utils.py` and remove its exports from `packages/aitraf-train/src/aitraf_train/utils/__init__.py`. No re-export shim (Principle VI). **Notebooks not repointed**: all three import from a defunct `aitraf.*` namespace predating the package split (`from aitraf.utils import ...`) and already raise `ModuleNotFoundError` on every cell, independently of this change. Repointing one import in a notebook whose other imports are equally dead would not make it runnable. Tracked as pre-existing breakage, not resolved here.
- [X] T014 [P] [US1] Implement `load_pose_artifact(path) -> PoseArtifact` in `packages/aitraf-api/src/aitraf_api/features/demo_predictions/render/poses.py` — `np.load(..., allow_pickle=True)`, validating all five rules in [data-model.md](./data-model.md) and raising a `PoseArtifactError` naming the specific violation. No coercion, no reshaping, no filling.
- [X] T015 [P] [US1] Implement `read_rotation_deg(path) -> int` in `packages/aitraf-api/src/aitraf_api/features/demo_predictions/render/video.py` using `ffprobe` via `subprocess`. **Raise when rotation metadata is absent** — do not default to 0 (research R-004; matches `get_video_rotation_deg`'s behavior).
- [X] T016 [US1] Implement `render_pose_video(source, artifact, destination)` in `packages/aitraf-api/src/aitraf_api/features/demo_predictions/render/video.py` — iterate frames with `imageio.v3`, apply `np.rot90(frame, k=read_rotation_deg(source) // 90)` **before drawing** (research R-004), draw each frame's detections onto a PIL canvas via `draw_pose_keypoints` with `POSE_DEFAULT_SKELETON`, and encode H.264 MP4 at the source framerate
- [X] T017 [US1] Raise in `render_pose_video` when the decoded frame count does not equal `len(artifact.frames)` — the artifact belongs to a different clip ([contracts/pose-artifact.md](./contracts/pose-artifact.md))

### Render helper tests

- [X] T018 [P] [US1] Test `load_pose_artifact` in `packages/aitraf-api/tests/features/demo_predictions/render/test_poses.py`: valid artifact, each missing key, non-dense `frames`, wrong keypoint trailing dims, length mismatch
- [X] T019 [US1] 🔴 Test rotation correctness in `packages/aitraf-api/tests/features/demo_predictions/render/test_video.py` — **the highest-risk behavior in this feature**. Build a fixture clip with non-zero rotation metadata plus a synthetic artifact whose keypoints sit at a known position, render, and assert the drawn pixels land at the expected coordinates in the output. A renderer that skips rotation must fail this test. Also cover: `read_rotation_deg` raising on absent metadata, and frame-count mismatch raising.
- [X] T020 [P] [US1] Test `render_pose_video` in `packages/aitraf-api/tests/features/demo_predictions/render/test_video.py` for a zero-detection frame — output must be written, frame preserved, no overlay, no error

### API publish path

- [X] T021 [US1] Add `derive_pose_key(video_id) -> str` returning `poses/{PurePosixPath(video_id).stem}.npz` in `packages/aitraf-api/src/aitraf_api/features/demo_predictions/clips.py` (research R-006)
- [X] T022 [US1] Replace `_ensure_public_video` in `packages/aitraf-api/src/aitraf_api/features/demo_predictions/clips.py`: keep the early return when the public object exists, then **remove the `copy_s3_object` call** and instead download the source clip, download the pose `.npz` from `derive_pose_key`, call `render_pose_video` into the temp directory, and upload the rendered file to `videos/{video_id}` with `ContentType="video/mp4"`. Raise `DemoClipError` when either source object is missing.
- [X] T023 [US1] Change `_ensure_public_thumbnail` to accept the **already-rendered local path** instead of `source_uri`, dropping its internal `download_s3_uri` call so `generate_thumbnail` runs against the rendered file. Delete the now-unused `_create_thumbnail` download branch.
- [X] T024 [US1] Restructure the per-row loop in `prepare_public_demo_clips` so one `tempfile.TemporaryDirectory` spans both assets — one download, one decode, both uploads — and both existence checks still short-circuit before any download. Update the counters and summary log to report rendered vs. reused.

### API tests

- [X] T025 [US1] Update `packages/aitraf-api/tests/features/demo_predictions/test_clips.py` for the new flow: first-time render+upload of both assets, both-exist reuse with zero downloads and zero renders, video-exists-but-thumbnail-missing, missing pose object raises, invalid pose artifact raises, missing source clip raises
- [X] T026 [P] [US1] Assert in `packages/aitraf-api/tests/features/demo_predictions/test_route.py` that the response shape is unchanged — same field names, no new fields, URLs carry no query string
- [X] T027 [P] [US1] Assert in `packages/aitraf-api/tests/test_app.py` that startup fails and readiness is not reported when pose data for a selected prediction is missing or invalid (spec FR-005)

---

## Phase 4: Rollout & End-to-End Validation (Stage 2)

**Purpose**: Prove it works against real infrastructure. Mostly checkpoints.

- [ ] T028 🧑 **CHECKPOINT — clear the stale public prefixes.** Existing objects under `videos/` and `thumbnails/` are non-pose, and preparation skips keys that already exist, so **without this the deploy silently keeps serving old assets and the feature looks broken**. Run the two `aws s3 rm --recursive` commands in [quickstart.md](./quickstart.md) §2. One-time operation, public bucket only.
- [ ] T029 🧑 **CHECKPOINT — cold start.** `task api:run`. **Verify**: prepare log reports a non-zero rendered count, readiness is reached, no errors. Record wall-clock time to readiness (research R-007).
- [ ] T030 🧑 **CHECKPOINT — response shape unchanged.** Call `/demo-predictions` ([quickstart.md](./quickstart.md) §4). **Verify**: exactly the previous fields, no additions, no query strings on URLs.
- [ ] T031 🧑 **CHECKPOINT — overlay is visible.** Fetch a `video_url` and its `thumbnail_url` unauthenticated ([quickstart.md](./quickstart.md) §5). **Verify**: skeleton visible in both, tracking the rider across the whole clip.
- [ ] T032 🧑 **CHECKPOINT — rotated clip specifically.** Repeat T031 on a clip with non-zero rotation metadata. **Verify**: the overlay lands on the rider, not offset or mirrored. Automated coverage exists in T019, but this is the failure that looks wrong rather than raising, so confirm it visually on real footage.
- [ ] T033 🧑 **CHECKPOINT — warm restart.** `docker compose restart api`. **Verify**: log reports all objects reused, zero renders, zero uploads; URLs captured in T031 still resolve.
- [ ] T034 🧑 **CHECKPOINT — failure scenarios.** Run all four rows of the [quickstart.md](./quickstart.md) §7 table (missing pose, corrupt pose, mismatched artifact, absent rotation metadata). **Verify**: each fails startup explicitly, and the listing never serves a non-pose asset.

---

## Phase 5: Polish

- [ ] T035 [P] Update `packages/aitraf-api` API documentation to state that published demo assets are pose-rendered
- [ ] T036 [P] Add a note to `specs/007-publish-demo-assets/contracts/public-demo-assets.md` pointing to [contracts/public-demo-assets.md](./contracts/public-demo-assets.md) as the current revision
- [X] T037 [P] Document the `upload_poses` step in `packages/aitraf-train/README.md` alongside the other `data_ops` steps (root README delegates pipeline docs there)
- [X] T038 Run `task lint` and `task format`; confirm no dead imports remain from the `draw_utils` move
- [ ] T039 Record the reproducibility values listed at the end of [quickstart.md](./quickstart.md) (run IDs, buckets, counts, timings, which rotated clip was checked) — spec VR-005

---

## Dependencies

```
Phase 1 (T001–T002)
    ↓
Phase 2 / Stage 1 (T003–T009)
    ↓
T010 🧑 poses in bucket ◄── HARD GATE: Stage 2 cannot start before this
    ↓
Phase 3 / Stage 2 (T011–T027)
    ↓
T028 🧑 clear public prefixes ◄── HARD GATE: skip this and T029–T033 give false passes
    ↓
Phase 4 (T029–T034 🧑)
    ↓
Phase 5 (T035–T039)
```

Within Phase 2: T003 → T004 → T006, T007; T005 parallel to T006/T007; T008/T009 after T007.

Within Phase 3: T011 → T012 → T013; T014/T015 parallel after T011; T016 needs T012+T014+T015; T017 after T016; T021 → T022 → T023 → T024; tests after their targets.

## Parallel Opportunities

- **T005** alongside T006/T007 (separate files)
- **T014 + T015** — different modules, both only need T011
- **T018 + T020** — different test targets
- **T026 + T027** — different test files
- **T035 + T036 + T037** — all documentation

## Implementation Strategy

**MVP = Stage 1 + Phase 3 + T028–T031.** That delivers the feature and proves it.
T032–T034 harden it; Phase 5 tidies up.

**Do not batch the checkpoints.** T010 and T028 are gates, not formalities:
running Stage 2 without T010 fails on every clip, and running Phase 4 without
T028 produces green checkpoints against stale non-pose assets — the worst
outcome, because it looks like success.

**Riskiest task is T019/T032 (rotation).** Everything else fails loudly. Rotation
failure renders a plausible-looking video with the skeleton in the wrong place,
which passes every automated check that does not specifically assert pixel
positions. Build the T019 fixture before writing T016 if you want the safety net
in place first.
