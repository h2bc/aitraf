# Contract: Public Demo Assets (revises spec 007)

Supersedes `specs/007-publish-demo-assets/contracts/public-demo-assets.md` on one
point only: **published asset content is now pose-rendered.** Keys, URLs,
signing, idempotency, and the route's zero-work property are unchanged.

## Layout

For `video_id = sample.mp4`:

| Asset | Key | Content |
|-------|-----|---------|
| Video | `videos/sample.mp4` | **Changed**: pose-rendered MP4 |
| Thumbnail | `thumbnails/sample.jpg` | **Changed**: still from the rendered video |

URL: `<AWS_ENDPOINT_URL>/<AITRAF_PUBLIC_ASSET_BUCKET>/<key>`, no query string.

## Preserved from 007

- Deterministic keys derived from `video_id` (FR-009)
- Stable, non-expiring, unsigned links (FR-008)
- Only the matched subset is published (FR-001)
- Idempotent: an existing object is reused, zero writes on restart (FR-006/007)
- No readiness with a broken asset (FR-011)
- Assets outside the selection are never deleted (FR-013)
- The route performs no existence checks, copies, uploads, or signing

## Changed from 007

| Aspect | 007 | Now |
|--------|-----|-----|
| Video production | `copy_s3_object` server-side, private → public | Download clip + poses, render locally, upload result |
| Thumbnail source | Separately downloaded source clip | The already-rendered local file |
| Downloads per clip | 1 (thumbnail path only) | 2 (clip + `.npz`), one decode total |
| New failure mode | — | Missing or invalid pose data blocks readiness |

## Rollout Requirement

Existing objects under `videos/` and `thumbnails/` are non-pose. Preparation
skips keys that already exist, so **those prefixes must be cleared once at
rollout** or stale non-pose assets will continue to be served.

This is a deliberate one-time operation on the public bucket only, and is the one
case where 007's FR-013 does not apply — the objects being removed are exactly
the selected subset being replaced, not unselected assets.

## Response

Unchanged. `DemoPrediction` keeps `video_url` and `thumbnail_url` with the same
names, types, and construction.
