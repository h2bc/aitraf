# Data Model: Pose-Rendered Demo Assets

## Pose Artifact (`.npz`)

Produced by `pose_and_bbox_extraction`, uploaded by `upload_poses`, consumed by
the API at startup. One file per source clip.

| Key | dtype | Shape | Constraint |
|-----|-------|-------|-----------|
| `frames` | `int32` | `(n_frames,)` | Dense, 0-based, strictly increasing |
| `keypoints` | `object` | `(n_frames,)` of `(n_det, 17, 2)` float | Normalized to `[0, 1]` against the **rotated** frame |
| `scores` | `object` | `(n_frames,)` of `(n_det, 17)` float | Per-keypoint confidence |

`n_det` is 0 or 1 under the current `max_det: 1` extraction config. A frame with
`n_det = 0` is valid and renders without an overlay.

**Loading**: requires `allow_pickle=True` because of the `object` arrays. See
research R-002 — these are trusted first-party artifacts read only from the
private bucket.

**Validation on load** (`load_pose_artifact`, raises on any failure):

1. All three keys present.
2. `frames` is 1-D, `int32`, dense and 0-based.
3. `keypoints` and `scores` have length equal to `frames`.
4. Every per-frame keypoint array has trailing dims `(17, 2)`.
5. Every per-frame score array has trailing dim `(17,)` and matching `n_det`.

No coercion, no reshaping, no filling of missing frames.

## Storage Layout

### Private bucket (`AWS_BUCKET`)

| Content | Key | Owner |
|---------|-----|-------|
| Source clip | existing key from `s3_path` | pre-existing |
| Pose artifact | `poses/{stem}.npz` | **new**, written by `upload_poses` |

`stem` is `PurePosixPath(video_id).stem`, matching how extraction names files on
disk and how thumbnails are already keyed.

### Public bucket (`AITRAF_PUBLIC_ASSET_BUCKET`)

| Content | Key | Change |
|---------|-----|--------|
| Video | `videos/{video_id}` | key unchanged; **content is now pose-rendered** |
| Thumbnail | `thumbnails/{stem}.jpg` | key unchanged; **content is now pose-rendered** |

No new prefix. Keys, URL construction, and the unsigned stable-link property from
spec 007 are all preserved.

**Migration consequence**: existing objects at these keys are non-pose. Because
preparation skips keys that already exist, a deploy against the current public
bucket will serve stale non-pose assets. The public bucket's `videos/` and
`thumbnails/` prefixes must be cleared once at rollout. This is recorded in
quickstart.md as a required step.

## Example

For `video_id = "clip_0042.mp4"`:

```
private  s3://$AWS_BUCKET/<existing clip key>
private  s3://$AWS_BUCKET/poses/clip_0042.npz
public   s3://$AITRAF_PUBLIC_ASSET_BUCKET/videos/clip_0042.mp4       (rendered)
public   s3://$AITRAF_PUBLIC_ASSET_BUCKET/thumbnails/clip_0042.jpg   (rendered)
```

Response, shape unchanged from today:

```
video_url      <AWS_ENDPOINT_URL>/<public bucket>/videos/clip_0042.mp4
thumbnail_url  <AWS_ENDPOINT_URL>/<public bucket>/thumbnails/clip_0042.jpg
```

## Rendered Video

| Property | Value |
|----------|-------|
| Container / codec | MP4 / H.264 |
| Frame rate | Same as source |
| Dimensions | Source dimensions after rotation is applied |
| Orientation | Rotation baked into pixels; output carries no rotation metadata |
| Overlay | `POSE_DEFAULT_SKELETON` (COCO-17, 16 edges), fixed repo-defined style |

Frame count must equal `len(frames)` in the pose artifact. A mismatch raises —
it means the artifact does not belong to this clip.

## Rendered Thumbnail

Unchanged extraction parameters (`-ss 0.5 -frames:v 1 -q:v 3`), but taken from
the **rendered** local file rather than a separately downloaded source. Same
source moment as before, now with the overlay visible.

## In-Memory Types

| Type | Fields | Notes |
|------|--------|-------|
| `PoseArtifact` | `frames: np.ndarray`, `keypoints: np.ndarray`, `scores: np.ndarray` | Frozen; returned by `load_pose_artifact` |
| `PoseUploadConfig` | `poses_dir: Path`, `prefix: str`, `force: bool` | Mirrors sibling `data_ops` configs; paths coerced in `__post_init__` |

The `DemoPrediction` response model is **not** modified.
