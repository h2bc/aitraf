# Implementation Plan: Pose-Rendered Demo Assets

**Branch**: `master` (no feature branch created) | **Date**: 2026-07-18 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/009-pose-rendered-assets/spec.md`

## Summary

The demo predictions listing must serve clips and thumbnails with the detected
skeleton drawn over the rider, replacing the current non-pose assets.

Approach: one new offline step uploads pose `.npz` artifacts to the private
bucket; the pose drawing helper moves from `aitraf-train` into the API's
`demo_predictions` feature alongside a new frame-rendering helper; and the API's startup publish step stops
server-side copying the source clip and instead downloads clip plus poses,
renders the overlay locally, grabs the thumbnail from that rendered file, and
uploads both to the public bucket under the existing keys.

The response schema is unchanged. Only what `video_url` and `thumbnail_url`
resolve to changes.

## Technical Context

**Language/Version**: Python 3.10+ (`>=3.10,<3.14`)

**Primary Dependencies**: Offline — Hydra, MLflow, ultralytics, boto3. Serving —
FastAPI, boto3, MLflow, Redis, plus new `numpy`, `pillow`, `imageio`, `av` on
`aitraf-api`. `ffmpeg`/`ffprobe` already present in the API image.
`aitraf-core` is unchanged.

**Storage**: Private `AWS_BUCKET` (source clips under existing keys, pose `.npz`
under a new `poses/` prefix); public `AITRAF_PUBLIC_ASSET_BUCKET` (`videos/`,
`thumbnails/`); local `storage/data/poses` on the training host.

**Testing**: `pytest` for the render helpers, pose loading, key derivation, and
the publish flow with a stubbed S3 client; command-level smoke validation for
cold start, warm restart, and the offline upload step.

**Target Platform**: Training — Linux CUDA host. Serving — `python:3.13-slim`
container, no GPU.

**Project Type**: ML pipeline plus a serving API

**Performance Goals**: Warm start unchanged (zero renders, zero uploads). Cold
start against an empty public bucket is bounded by the matched demo subset size;
see R-007. Request-path latency unchanged — no work moves into request handling.

**Constraints**: The API image must not gain `torch`, `ultralytics`, or
`aitraf-ml-core`. Rendering must reproduce extraction's rotation handling exactly
(R-004). Preparation stays idempotent and fails loudly.

**Scale/Scope**: The matched classification ∩ AQA demo subset only.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Initial evaluation — **PASS**. (Planning expected a Package By Feature
justification for promoting the helper into `aitraf-core`; implementation dropped
that promotion, so no justification is needed — see research R-001.)

- **No Excessive Fallbacks**: PASS. Missing, unreadable, or shape-mismatched pose
  data raises and blocks readiness. Absent rotation metadata raises rather than
  defaulting to 0. `upload_poses` raises on first failure rather than counting
  failures (R-008). No fallback to the non-pose asset exists, because that path
  is removed.
- **Package By Feature**: PASS with justification. Serving behavior stays in the
  `demo_predictions` feature; the offline step stays in `aitraf-train/data_ops`.
  Rendering is owned by the `demo_predictions` feature rather than promoted to a
  shared package, because the API is its only production consumer. No package
  boundary changes at all; `aitraf-core` keeps its enforced boto3-only surface.
- **Function Decomposition**: PASS. Work decomposes into `load_pose_artifact`,
  `read_rotation_deg`, `render_pose_video`, `derive_pose_key`,
  `upload_directory`, and a modified `_ensure_public_video`. Each has explicit
  inputs, outputs, and failure modes.
- **Functional Programming And State**: PASS. Pose loading, key derivation, and
  frame drawing are pure. Mutable state is confined to the frame writer and the
  temporary directory, both at the I/O boundary.
- **Reproducibility**: PASS. The offline step is Hydra-configured like every
  other `data_ops` step. Rendering is deterministic given clip, `.npz`, and the
  fixed overlay style. Pinned run IDs and bucket configuration are unchanged.
- **No Legacy Compatibility**: PASS. The `copy_s3_object` path in
  `_ensure_public_video` is removed, not flagged. `draw_utils.py` is deleted from
  `aitraf-train` and its exports removed from `utils/__init__.py`. No dual-asset
  mode, no opt-out flag.
- **Required Types Over Defensive Normalization**: PASS. One accepted `.npz`
  shape (R-002), validated on load and rejected otherwise. No scalar-or-list or
  path-or-dict variants introduced.

Post-Phase-1 re-check: **PASS, unchanged.** The design added no new fallback
branch, no parallel structure, and no alternate accepted shape.

## Project Structure

### Documentation (this feature)

```text
specs/009-pose-rendered-assets/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   ├── pose-artifact.md
│   └── public-demo-assets.md
├── checklists/
│   └── requirements.md
└── tasks.md             # Created by /speckit-tasks, not this command
```

### Source Code (repository root)

```text
packages/
├── aitraf-core/                            # UNCHANGED (boto3-only boundary enforced)
├── aitraf-api/
│   ├── pyproject.toml                      # + numpy, pillow, imageio, av
│   ├── Dockerfile                          # unchanged (ffmpeg already present)
│   └── src/aitraf_api/features/demo_predictions/
│       ├── render/
│       │   ├── __init__.py                 # NEW
│       │   ├── draw.py                     # NEW: moved from aitraf-train
│       │   ├── poses.py                    # NEW: load + validate .npz
│       │   └── video.py                    # NEW: rotation + render loop
│       ├── clips.py                        # MODIFIED: copy -> download+render+upload
│       └── thumbnails.py                   # unchanged
└── aitraf-train/
    ├── configs/data_ops.yaml               # + upload_poses block
    ├── scripts/data_ops_pipeline.py        # + upload_poses clause
    └── src/aitraf_train/
        ├── storage/artifacts.py            # NEW: upload_directory()
        ├── preparation/
        │   ├── data_ops/upload_poses.py    # NEW
        │   └── label_ops/upload_pairs.py   # MODIFIED: onto upload_directory()
        └── utils/draw_utils.py             # DELETED: moved into aitraf-api
```

**Structure Decision**: Offline behavior stays in `aitraf-train/data_ops`
alongside its sibling steps. Rendering is owned by the `demo_predictions` feature
in `aitraf-api`, since the API is its only production consumer (the previous
`draw_utils` had no callers outside notebooks). `aitraf-core` keeps its enforced
boto3-only boundary. No new top-level package, no parallel publish pipeline.

## Implementation Phases

### Phase A — Offline pose upload

1. Add `upload_directory()` to `aitraf_train/storage/artifacts.py` — walk, skip
   existing keys unless `force`, upload, log progress, raise on first failure.
2. Refactor `upload_pairs` onto it. Behavior change: raises on failure now.
3. Add `upload_poses` (`PoseUploadConfig(poses_dir, prefix, force)`).
4. Add the `upload_poses` block to `data_ops.yaml` and its `enabled` clause to
   `data_ops_pipeline.py`, ordered after `pose_and_bbox_extraction`.

Independently verifiable: run the pipeline, confirm `.npz` objects land under
`poses/`, re-run and confirm all skipped.

### Phase B — Render helpers in `aitraf-api`

5. Move `draw_utils.py` to `demo_predictions/render/draw.py`; delete the original;
   clear `aitraf_train/utils/__init__.py` and repoint the notebook imports.
6. Add `render/poses.py` — `load_pose_artifact(path)` validating the R-002 shape
   and raising on mismatch.
7. Add `render/video.py` — `read_rotation_deg(path)` via `ffprobe` and
   `render_pose_video(source, poses, destination)` applying the R-004 rotation,
   decoding through the pinned PyAV plugin (R-011), drawing per frame, streaming
   H.264 MP4 output.
8. Add `numpy`, `pillow`, `imageio`, `av` to `aitraf-api`.

Independently verifiable: unit tests on a short fixture clip plus a synthetic
`.npz`, including a rotated clip and a zero-detection frame.

### Phase C — API publish path

9. Add `derive_pose_key(video_id)` → `poses/{stem}.npz`.
10. Replace `_ensure_public_video`'s `copy_s3_object` with download clip →
    download `.npz` → render → upload rendered MP4 to `videos/{video_id}`.
11. Rewire `_ensure_public_thumbnail` to take the already-rendered local file
    instead of re-downloading the source, so both assets come from one download
    and one decode.
12. Restructure the per-row loop so the shared temporary directory spans both
    assets.

Independently verifiable: cold start against an empty public bucket, then warm
restart proving zero work.

### Phase D — Validation and docs

13. Tests per VR-001.
14. Smoke validation per VR-002 and the visual check in VR-003.
15. Update API docs and `specs/007-publish-demo-assets/contracts/` to record that
    published assets are now pose-rendered.

## Risks

| Risk | Mitigation |
|------|-----------|
| Overlay misaligned on rotated clips — fails silently, looks wrong rather than erroring | R-004 mandates matching extraction's `np.rot90`; VR-003 requires visual confirmation on a rotated clip specifically |
| Cold-start render time grows with the demo subset | Idempotent, so paid once; escape hatch is moving rendering offline with no contract change (R-007) |
| Decoder plugin disagreement silently misaligns every overlay | `plugin="pyav"` pinned explicitly in `video.py`; two tests assert rendered pixel positions and both fail if rotation is removed (research R-011) |
| `allow_pickle=True` on `.npz` load | Artifacts are produced by this repo and read only from the private bucket; documented in R-002 as a constraint not to relax |
| Refactor changes `upload_pairs` failure behavior beyond this feature's scope | Called out explicitly in R-008 so it is reviewed deliberately |

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None. The `aitraf-core` promotion contemplated during planning was dropped once `test_dependency_boundary.py` was found; rendering is feature-local in `aitraf-api`, which is what AGENTS.md prescribes for a single-consumer helper | — | — |
