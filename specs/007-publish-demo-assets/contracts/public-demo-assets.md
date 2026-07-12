# Public Demo Asset Contract

## Runtime Configuration

The API requires these storage variables:

| Variable | Meaning |
|----------|---------|
| `AWS_ENDPOINT_URL` | Shared S3-compatible server and public HTTP origin |
| `AWS_DEFAULT_REGION` | Existing client region |
| `AWS_ACCESS_KEY_ID` | Authenticated source/public-bucket access key |
| `AWS_SECRET_ACCESS_KEY` | Authenticated source/public-bucket secret |
| `AWS_BUCKET` | Private source bucket |
| `AITRAF_PUBLIC_ASSET_BUCKET` | Publicly downloadable destination bucket |

No media URL base, signing expiration, or public-bucket credential override is
accepted. Missing required variables fail API startup.

## Public Layout

For a selected `video_id` of `sample.mp4`:

| Asset | Object key | Public URL |
|-------|------------|------------|
| Video | `videos/sample.mp4` | `<AWS_ENDPOINT_URL>/<AITRAF_PUBLIC_ASSET_BUCKET>/videos/sample.mp4` |
| Thumbnail | `thumbnails/sample.jpg` | `<AWS_ENDPOINT_URL>/<AITRAF_PUBLIC_ASSET_BUCKET>/thumbnails/sample.jpg` |

Endpoint trailing slashes are normalized once. Bucket and key segments are
percent-encoded without encoding key separators. URLs contain no query string,
signature, expiry, or authentication material.

## Startup Contract

1. Download and match classification and AQA prediction rows.
2. Validate and deduplicate the selected media set.
3. Reuse each existing public video; otherwise copy the missing video from its
   private source.
4. Reuse each existing public thumbnail; otherwise
   download the source video temporarily, generate the thumbnail, and upload it.
5. Attach stable public video and thumbnail URLs to the prepared classification
   rows.
6. Construct the application only after every selected asset is valid.

The workflow never publishes an unmatched row, never overwrites a conflicting
object as normal behavior, and never deletes an unselected object.

## Response Contract

Authenticated `GET /demo-predictions` retains the existing response shape.
`video_url` and `thumbnail_url` now contain prepared stable public URLs. Repeated
requests return identical URLs while startup inputs remain unchanged.

The route performs no object existence checks, copies, uploads, downloads,
thumbnail generation, or URL signing.

## Failure Contract

Application startup fails with the affected asset identity when:

- public bucket configuration is missing or invalid;
- source and public buckets are the same;
- a row has an invalid ID or source URI;
- the authenticated client cannot inspect either bucket;
- a source object is absent;
- a copy, download, thumbnail generation, or upload fails;
- duplicate rows map one public identity to different sources.

No private or signed URL is returned as a fallback.

## Removed API Contract

The API no longer provides or retains:

- an asset URL presigner protocol or factory;
- a media URL expiration constant;
- presigner state on the application;
- a route/service presigner callback;
- per-request URL transformation;
- signed video or thumbnail URL tests and documentation.

The generic core presigning helper used by offline training reports is not part
of the API contract and is unchanged.
