# Validation: Publish Demo Assets

**Date**: 2026-07-11

## Automated Tests

```text
uv run pytest packages/aitraf-core/tests packages/aitraf-api/tests -q
37 passed in 1.45s
```

Source compilation also completed successfully:

```text
uv run python -m compileall -q packages/aitraf-core/src packages/aitraf-api/src
```

## API Presigning Removal

The required scan returned no matches under `packages/aitraf-api`:

```text
rg -n 'create_asset_url_presigner|AssetUrlPresigner|VIDEO_URL_EXPIRATION_SECONDS|presign_asset_url|presigned' packages/aitraf-api
```

The generic core presigner remains for the existing offline training metrics
consumer and is not imported by the API.

## Configured Storage Smoke Validation

- Both configured prediction artifacts downloaded successfully.
- Ten matched demo rows were prepared.
- Missing selected videos and thumbnails were published at deterministic keys.
- Two consecutive application constructions returned identical stable URLs.
- URLs contain no query strings or signing data.

## Remaining External Failure

Anonymous HTTP GET of a generated public video URL returned `403 Forbidden`.
The bucket's anonymous download policy is not currently effective for the
derived path-style URL. Task T029 remains incomplete until anonymous GET works
for every returned video and thumbnail.

No credentials, tokens, bucket names, or run identifiers are recorded here.
