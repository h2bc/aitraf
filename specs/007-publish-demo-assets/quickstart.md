# Quickstart: Validate Public Demo Assets

## Prerequisites

- The private source and public destination buckets exist on the same configured
  S3-compatible endpoint.
- Current AWS credentials can read the source bucket and inspect/write the
  public bucket.
- The public bucket policy permits anonymous object downloads.
- Classification and AQA run IDs reference valid matching prediction artifacts.
- FFmpeg is available for missing-thumbnail generation.

## Configure

Set all existing API variables and the new required destination:

```bash
export AWS_ENDPOINT_URL=https://storage.h2bcweb.com
export AWS_DEFAULT_REGION=auto
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_BUCKET=aitraf
export AITRAF_PUBLIC_ASSET_BUCKET=...
export AITRAF_API_TOKEN=...
export AITRAF_CLASSIFICATION_PREDICTIONS_RUN_ID=...
export AITRAF_AQA_PREDICTIONS_RUN_ID=...
```

The public URL is derived from the endpoint and public bucket. Do not configure
a signing expiry or separate media base URL.

## Automated Validation

Run focused package tests:

```bash
uv run pytest packages/aitraf-core/tests packages/aitraf-api/tests -q
```

Prove the API implementation no longer contains presigning behavior:

```bash
rg -n 'create_asset_url_presigner|AssetUrlPresigner|VIDEO_URL_EXPIRATION_SECONDS|presign_asset_url|presigned' packages/aitraf-api
```

The source scan must return no runtime, test, or documentation matches. The
generic core presigner may remain because offline training metrics still consume
it.

Automated tests must demonstrate:

- matched-subset-only publication;
- existing-object reuse with zero writes;
- one video copy when only a public video is missing;
- one source download, thumbnail generation, and upload when only a thumbnail
  is missing;
- identical URLs and zero writes on a second preparation run;
- explicit failure for invalid config, missing source, storage operations,
  and thumbnail generation;
- no storage calls during repeated route requests.

## Runtime Smoke Test

Start the API:

```bash
task api:run
```

Startup must complete publication before readiness. Request predictions and save
the payload:

```bash
curl --fail \
  -H "Authorization: Bearer ${AITRAF_API_TOKEN}" \
  http://localhost:8000/demo-predictions \
  --output /tmp/aitraf-demo-predictions.json
```

For every `video_url` and `thumbnail_url`, verify:

- it starts with
  `${AWS_ENDPOINT_URL%/}/${AITRAF_PUBLIC_ASSET_BUCKET}/`;
- it has no query string, signature, expiry, or credential material;
- `curl --fail --location "<asset-url>" --output /dev/null` succeeds without AWS
  headers or credentials.

Use anonymous GET rather than relying only on HEAD because public policies may
grant these operations differently.

## Idempotent Restart

Stop and restart the API with exactly the same configuration and run IDs.
Expected outcomes:

- startup reports every selected public object as reused;
- it performs no copies or uploads;
- the response payload contains byte-for-byte identical media URLs;
- previously published objects outside the current subset remain untouched.

Then add one new matched prediction asset in a controlled validation input. Only
that missing public video and thumbnail may be created.
