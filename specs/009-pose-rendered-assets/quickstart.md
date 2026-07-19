# Quickstart: Validating Pose-Rendered Demo Assets

## Prerequisites

- Training host with `storage/data/clips` and `storage/data/poses` populated
- `.env` with `AWS_ENDPOINT_URL`, `AWS_BUCKET`, `AITRAF_PUBLIC_ASSET_BUCKET`,
  `AITRAF_API_TOKEN`, both prediction run IDs, `AITRAF_REDIS_URL`
- `ffmpeg` and `ffprobe` on PATH (already in the API image)

## 1. Upload pose artifacts

The `data_ops` config is flat — overrides take no `data_ops.` prefix.

```bash
uv run python packages/aitraf-train/scripts/data_ops_pipeline.py \
  download_labels.enabled=false \
  download_clips.enabled=false \
  pose_and_bbox_extraction.enabled=false \
  video_mae_feature_extraction.enabled=false \
  download_pairwise_labels.enabled=false \
  upload_poses.enabled=true
```

**Expected**: summary logs `N uploaded, 0 skipped`.

Re-run the same command. **Expected**: `0 uploaded, N skipped` — idempotency.

Verify a key exists:

```bash
aws --endpoint-url "$AWS_ENDPOINT_URL" s3 ls "s3://$AWS_BUCKET/poses/" | head
```

## 2. Clear stale public assets (one time, at rollout only)

Required — see [contracts/public-demo-assets.md](./contracts/public-demo-assets.md).
Existing objects are non-pose and would be reused rather than replaced.

```bash
aws --endpoint-url "$AWS_ENDPOINT_URL" s3 rm "s3://$AITRAF_PUBLIC_ASSET_BUCKET/videos/" --recursive
aws --endpoint-url "$AWS_ENDPOINT_URL" s3 rm "s3://$AITRAF_PUBLIC_ASSET_BUCKET/thumbnails/" --recursive
```

## 3. Cold start

```bash
docker compose up --build api
```

**Expected**: log line reporting assets prepared with a non-zero rendered count,
readiness reached, no errors. Note the wall-clock time to readiness — this is the
cold-start cost from research R-007.

## 4. Verify the response shape is unchanged

```bash
curl -s -H "Authorization: Bearer $AITRAF_API_TOKEN" \
  "http://127.0.0.1:${AITRAF_API_PORT:-8001}/demo-predictions" | head -c 800
```

**Expected**: each entry still has exactly `video_id`, `video_url`,
`thumbnail_url`, `person`, `key_foot`, `ground_truth`, `predictions`. No new
fields. URLs carry no query string.

## 5. Verify assets are pose-rendered

```bash
VIDEO_URL=$(curl -s -H "Authorization: Bearer $AITRAF_API_TOKEN" \
  "http://127.0.0.1:${AITRAF_API_PORT:-8001}/demo-predictions" \
  | python -c "import sys,json; print(json.load(sys.stdin)[0]['video_url'])")

curl -sfo /tmp/posed.mp4 "$VIDEO_URL" && echo "fetched unauthenticated: ok"
ffprobe -v error -show_entries stream=width,height,nb_frames /tmp/posed.mp4
```

Open `/tmp/posed.mp4` and the thumbnail URL. **Expected**: skeleton visible in
both, tracking the rider across the full clip.

**Include at least one rotated clip** (research R-004 — misalignment there fails
visually, not loudly).

## 6. Warm restart — idempotency

```bash
docker compose restart api
```

**Expected**: readiness reached quickly, log reports all objects reused, zero
renders and zero uploads. Previously captured URLs still resolve.

## 7. Failure validation

| Scenario | Setup | Expected |
|----------|-------|----------|
| Missing pose data | Delete one `poses/*.npz` for a selected prediction, clear its public objects, restart | Startup fails explicitly; no non-pose asset published |
| Corrupt pose data | Upload an `.npz` missing the `keypoints` key, clear public objects, restart | Explicit validation error naming the key |
| Mismatched artifact | Swap one clip's `.npz` for another clip's, clear public objects, restart | Frame-count mismatch raises |
| No rotation metadata | Point at a clip stripped of rotation metadata | Explicit error, no default to 0° |

In every case: readiness must **not** be reported, and the listing must never
serve a non-pose asset.

## Record for reproducibility

- Both prediction run IDs
- `AWS_BUCKET`, `AITRAF_PUBLIC_ASSET_BUCKET`, `AWS_ENDPOINT_URL`
- Count of `.npz` uploaded, clips rendered
- Cold-start and warm-start times to readiness
- Which rotated clip was visually confirmed
