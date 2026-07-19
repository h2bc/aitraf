# Phase 0 Research: Pose-Rendered Demo Assets

All open decisions from planning input are resolved below against the current
code. No `NEEDS CLARIFICATION` remains.

## R-001: Which package owns the drawing and render helper

**Decision (revised during implementation)**:
`packages/aitraf-api/src/aitraf_api/features/demo_predictions/render/`.
`aitraf-api` gains `numpy`, `pillow`, `imageio`, and `av`. **`aitraf-core` is not
touched.**

**Why the original `aitraf-core` decision was wrong**: planning missed
`packages/aitraf-core/tests/test_dependency_boundary.py`, an enforced guard that
asserts `dependencies == ["boto3"]` exactly and lists `numpy` among prohibited
imports. Moving the render module into `aitraf-core` would have failed both
assertions, and deleting a deliberate architectural guard to accommodate a design
is not an acceptable resolution.

**Why `aitraf-api` is the correct home anyway**: a grep of every consumer shows
`draw_utils` is imported *only* by notebooks — no production code in
`aitraf-train` uses it. So there are not two feature surfaces needing it; there is
one, the API. AGENTS.md says to promote to shared packages only when multiple
surfaces genuinely need it, so feature-local ownership is what the conventions
actually call for.

`aitraf-ml-core` remains disqualified (`torch==2.9.1`, `torchcodec`, `kornia`,
`transformers`), and it is not copied into the API image regardless.

**Consequence**: the notebooks that used `aitraf_train.utils` must import from
`aitraf_api.features.demo_predictions.render` instead.

## R-002: Pose artifact payload shape

**Decision**: The required shape is the three-array payload written by
`_prepare_results` in
`packages/aitraf-train/src/aitraf_train/preparation/data_ops/pose_and_bbox_extraction.py`:

| Key | dtype | Shape | Meaning |
|-----|-------|-------|---------|
| `frames` | `int32` | `(n_frames,)` | Frame indices, 0-based, dense |
| `keypoints` | `object` | `(n_frames,)` of `(n_det, 17, 2)` float | Normalized `xyn` coordinates in `[0, 1]` |
| `scores` | `object` | `(n_frames,)` of `(n_det, 17)` float | Per-keypoint confidence |

**Rationale**: Read directly from the producing code. Keypoints come from
`kpts.xyn`, which is already normalized — matching what `draw_pose_keypoints`
expects, since it multiplies by `width`/`height` itself. No coordinate conversion
is needed.

**Consequence**: `keypoints` and `scores` are `dtype=object`, so `np.load` must
be called with `allow_pickle=True`. Pose artifacts are therefore trusted inputs
produced by this repository's own pipeline and read only from the private bucket.
This must not be relaxed to accept externally supplied `.npz` files.

**Validation on load**: reject any file missing one of the three keys, whose
`frames` is not dense and 0-based, or whose per-frame keypoint arrays do not have
trailing dimensions `(17, 2)`. One required shape, no coercion (Principle VII).

## R-003: Multi-person selection rule

**Decision**: Draw every detection present in the frame. No selection or ranking
logic.

**Rationale**: `data_ops.yaml` sets `pose_and_bbox_extraction.max_det: 1`, so
`n_det` is 0 or 1 in practice and a selection rule would be dead code. Drawing
all detections is correct for `n_det = 1`, degrades sensibly if `max_det` is ever
raised, and adds no branches.

**Frames with no detection**: `n_det = 0` yields an empty array, the draw loop
iterates zero times, and the frame is written unmodified. This satisfies the
spec's no-detection edge case with no special-casing.

**Alternatives considered**: highest-mean-confidence selection, largest-box
selection (both rejected: unreachable under `max_det: 1`).

## R-004: Rotation handling

**Decision**: The renderer MUST read rotation with `ffprobe` and apply the same
`np.rot90(frame, k=rotation_deg // 90)` that extraction applied, before drawing.

**Rationale**: This is the highest-risk correctness detail in the feature.
`_iter_frames` rotates each frame *before* running YOLO, so the stored normalized
keypoints are relative to the **rotated** frame, not the stored frame. Rendering
onto unrotated frames would silently place the skeleton in the wrong position on
every rotated clip — a wrong-looking demo rather than an error.

**Why ffprobe and not `av`**: extraction gets rotation from
`get_video_rotation_deg` in `aitraf_ml_core/processing/video.py`, which uses `av`.
The API cannot import `aitraf-ml-core` (R-001), and `av` is a compiled wheel
bundling its own FFmpeg. `ffprobe` ships with the `ffmpeg` package already
installed in the API image, so it costs nothing.

**Note**: `get_video_rotation_deg` raises when rotation metadata is absent, so
clips reaching extraction always have it. The renderer MUST match that behavior
and raise rather than defaulting to 0 (Principle I).

## R-005: Frame decode and video encode on the API

**Decision (revised during implementation)**: `imageio.v3` with the decoder
**explicitly pinned to the PyAV plugin** for both read and write. Adds `imageio`
and `av`. `imageio-ffmpeg` is *not* used — see R-011.

**Rationale**: symmetry with the producing code keeps rotation and frame ordering
aligned, which R-004 identifies as the main risk. Pinning the plugin rather than
relying on `imageio`'s auto-selection is load-bearing: the plugin choice depends
on what happens to be installed, and the two plugins disagree about frame counts
(R-011). PyAV also writes video, so the ~25MB bundled `imageio-ffmpeg` binary is
unnecessary.

**Output encoding**: H.264 in MP4, preserving source framerate, written frame by
frame rather than buffering the clip in memory (a 1080×1920 clip is ~770MB
uncompressed).

## R-011: Decoder frame-count mismatch (discovered during implementation)

**Finding**: `imageio`'s FFMPEG plugin and its PyAV plugin report different frame
counts for these clips. Measured on eight clips, the FFMPEG plugin returns 6–9
*more* frames than PyAV. `ffprobe -count_frames` agrees with PyAV, and PyAV's
count matches the pose artifacts exactly.

| Decoder | Frames (sample clip) | Matches artifact |
|---------|---------------------|------------------|
| `ffprobe -count_frames` | 124 | yes |
| `imageio` FFMPEG plugin | 131 | no |
| `imageio` PyAV plugin | 124 | yes |
| PyAV directly | 124 | yes |

**Decision**: pin `plugin="pyav"` for decoding. Extraction's
`iio.imiter(str(clip_path))` resolved to PyAV in the training environment, which
is why the artifacts carry PyAV's frame count.

**Why this matters more than it looks**: the two plugins also disagree about
rotation. The FFMPEG plugin auto-applies container rotation and returns
display-oriented frames; PyAV returns stored-orientation frames and leaves
rotation to the caller. Decoding with the FFMPEG plugin therefore breaks the
overlay twice over — frame indices drift, *and* extraction's `np.rot90` becomes a
double rotation. Both failures are silent: the output is a valid, plausible video
with the skeleton in the wrong place.

**Verified**: rendering a real clip with PyAV + `np.rot90(k=rotation // 90)`
places the skeleton exactly on the rider. The same clip decoded with the FFMPEG
plugin puts it on a blank wall.

**Follow-up worth considering** (not part of this feature): extraction relies on
`imageio`'s implicit plugin selection. If a future environment resolves it to the
FFMPEG plugin, artifacts would be produced against double-rotated frames with no
error raised. Pinning the plugin in `pose_and_bbox_extraction.py` too would close
that hole.

## R-006: Pose key derivation

**Decision**: `poses/{PurePosixPath(video_id).stem}.npz` in the private
`AWS_BUCKET`.

**Rationale**: Extraction writes `poses_dir / f"{clip.stem}.npz"`, so stem-based
naming is already the convention on disk, and `upload_poses` preserves it. It
also matches the existing thumbnail key derivation in
`packages/aitraf-api/src/aitraf_api/features/demo_predictions/clips.py:47`
(`thumbnails/{stem}.jpg`), so the API gains no new naming concept.

`_clip_values` already validates that `video_id` is a bare filename with a
suffix, so the stem is well-defined at that point.

## R-007: Startup cost and readiness

**Decision**: Accept synchronous rendering during startup. Do not add background
preparation or a readiness-deferral mechanism.

**Rationale**: Preparation is idempotent and keyed on public-object existence, so
rendering happens once per clip for the lifetime of the bucket. Warm starts do
zero work and are unchanged from today. Only a cold start against an empty public
bucket pays the cost, and it is bounded by the size of the matched demo subset.

**Risk accepted**: A cold start now decodes and re-encodes each clip rather than
issuing a server-side `copy_s3_object`. If the demo subset grows large enough
that cold-start time becomes a deployment problem, moving rendering offline into
`aitraf-train` is the escape hatch — it requires no API contract change, only a
change to what the API copies.

## R-008: `upload_poses` failure behavior

**Decision**: Raise on the first upload failure. Do not replicate the
warn-and-count behavior of `upload_pairs`.

**Rationale**: `packages/aitraf-train/src/aitraf_train/preparation/label_ops/upload_pairs.py:79-83`
logs per-file failures and only raises when nothing at all uploaded. For poses,
a silently skipped file surfaces much later as an API startup failure, far from
its cause. Principle I requires breaking loudly at the source.

**Scope note**: the shared helper introduced for this (R-009) changes
`upload_pairs` to the same raise-on-failure behavior. That is a deliberate
behavior change to an existing step, recorded here because it is a side effect of
the refactor rather than a requirement of this feature.

## R-009: Shared upload helper

**Decision**: Extract `upload_directory(directory, *, bucket, prefix, force,
s3_client) -> int` into `aitraf_train/storage/artifacts.py`. Both `upload_pairs`
and `upload_poses` call it.

**Rationale**: A copied `upload_pairs` would duplicate the walk, existence-skip,
progress-logging, and summary loop nearly verbatim. AGENTS.md and Principle III
require extraction over duplication. The helper stays in `aitraf-train` because
only offline steps upload; promoting it to `aitraf-core` would be speculative.

## R-010: Boxes are out of scope

**Decision**: `upload_poses` uploads pose `.npz` only. Box artifacts stay local.

**Rationale**: The overlay draws skeletons, not boxes. Uploading boxes would be
unused data. This differs from the earlier brainstorm position, which paired them
so that a future `download_poses` step would not trip the
`pose_out.exists() and box_out.exists()` skip condition in extraction — but with
no `download_poses` step in this feature, that condition is never consulted
against remote state and the pairing buys nothing.

**Follow-up**: if `download_poses` is added later, it must either sync boxes too
or change that skip condition. Recorded so the trap is not rediscovered.
