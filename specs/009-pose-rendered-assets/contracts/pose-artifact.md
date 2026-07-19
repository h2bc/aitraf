# Contract: Pose Artifact Exchange

Defines the boundary between the offline extraction pipeline
(`packages/aitraf-train`) and the serving renderer (`packages/aitraf-api` via
`packages/aitraf-core`).

## Producer

`pose_and_bbox_extraction` writes `poses_dir/<stem>.npz` via
`np.savez_compressed`. Unchanged by this feature.

`upload_poses` uploads those files to `s3://$AWS_BUCKET/poses/<stem>.npz`.

- Skips keys that already exist unless `force: true`
- Raises on the first upload failure
- Uploads pose artifacts only; box artifacts stay local (research R-010)

## Consumer

The API derives `poses/{PurePosixPath(video_id).stem}.npz` and downloads it
during startup preparation, for matched demo predictions only.

## Payload

| Key | dtype | Shape |
|-----|-------|-------|
| `frames` | `int32` | `(n_frames,)` |
| `keypoints` | `object` | `(n_frames,)` of `(n_det, 17, 2)` |
| `scores` | `object` | `(n_frames,)` of `(n_det, 17)` |

Coordinates are normalized to `[0, 1]` against the frame **after** rotation is
applied. This is the single most important property of the contract: the
consumer must rotate frames before drawing, or every rotated clip is silently
misaligned.

### Decoder requirement

Both sides MUST decode through **PyAV**. `imageio`'s FFMPEG plugin reports 6–9
more frames per clip than PyAV and auto-applies container rotation, so decoding
with it breaks the contract twice: frame indices drift out of alignment, and the
caller's rotation becomes a double rotation. `ffprobe -count_frames` agrees with
PyAV, and PyAV's count is what the artifacts were produced against.

Frame index `i` in the artifact corresponds to the `i`-th frame yielded by a PyAV
decode of the source clip. Any other decoder invalidates that correspondence
silently — the render succeeds and the output looks plausible.

## Required Failures

The consumer MUST raise, and MUST NOT fall back to publishing a non-pose asset,
when:

| Condition | Reason |
|-----------|--------|
| Object missing at the derived key | Demo prediction has no pose data |
| Any of the three keys absent | Not a pose artifact |
| `frames` not dense 0-based `int32` | Unusable indexing |
| Per-frame keypoints not trailing `(17, 2)` | Wrong keypoint model |
| `len(keypoints) != len(frames)` | Corrupt artifact |
| Frame count mismatch vs. decoded clip | Artifact belongs to a different clip |
| Source clip has no rotation metadata | Extraction required it; renderer must too |

## Explicitly Valid

| Condition | Behavior |
|-----------|----------|
| `n_det == 0` for a frame | Render that frame with no overlay; continue |
| `n_det > 1` for a frame | Draw every detection; no selection logic |
| Low confidence scores | Draw regardless; `scores` is carried but not thresholded |

## Stability

The key layout `poses/<stem>.npz` and the three-array payload are the contract.
Changing either requires re-uploading artifacts and updating both sides in the
same change — no dual-read bridge (Principle VI).
