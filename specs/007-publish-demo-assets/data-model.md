# Data Model: Publish Demo Assets

## Prediction Asset Row

The classification prediction row is the authoritative media source after
classification and AQA rows are matched.

Required media fields:

- `video_id`: non-empty filename basename with a suffix.
- `s3_path`: canonical private source URI whose bucket equals `AWS_BUCKET`.

The remaining existing display, ground-truth, and prediction fields retain their
current required schema. Alternate media field names and artifact-provided
thumbnail paths are not accepted.

Validation rules:

- `video_id` contains no directory, parent, or absolute path components.
- Source URI has the `s3` scheme, a non-empty key, and the configured source
  bucket.
- One public video or thumbnail key maps to exactly one source video.
- All selected classification rows have a matching AQA row before publication.

## Public Asset Configuration

Required fields:

- `endpoint_url`: existing absolute HTTP(S) storage endpoint from
  `AWS_ENDPOINT_URL`.
- `source_bucket`: existing private bucket from `AWS_BUCKET`.
- `public_bucket`: public destination from `AITRAF_PUBLIC_ASSET_BUCKET`.

The source and public buckets must be non-empty and distinct. Region and
credentials remain part of the existing storage-client configuration rather
than this API-specific entity.

## Selected Demo Asset

Derived immutable fields:

- `video_id`
- `source_uri`
- `source_bucket`
- `source_key`
- `public_video_key`: `videos/<video_id>`
- `public_thumbnail_key`: `thumbnails/<video-stem>.jpg`
- `video_url`
- `thumbnail_url`

Relationship: exactly one selected demo asset corresponds to one matched
classification/AQA prediction pair. Duplicate identical references collapse to
one publication decision; conflicting mappings fail before writes.

## Prepared Prediction Row

The served classification row after successful preparation contains all current
prediction/display fields plus:

- `video_url`: stable public URL for the published video.
- `thumbnail_url`: stable public URL for the published thumbnail.

It no longer receives or depends on `thumbnail_s3_path`. The service consumes
the two URL fields directly; no callback transforms storage URIs during a
request.

## State Transitions

```text
matched prediction row
        |
        v
selected asset --invalid/missing source--> startup failure
        |
        v
inspect public video ----missing----> copy from private source
        |
        v
inspect public thumbnail -missing---> download source, generate, upload
        |
        v
validate both objects and anonymous URL identities
        |
        v
prepared prediction row --> application ready
```

Existing valid objects transition directly through inspection without writes.
Previously published but unselected objects have no transition and are left
unchanged.
