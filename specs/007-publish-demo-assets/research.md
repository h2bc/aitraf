# Research: Publish Demo Assets

## API-Owned Startup Publication

**Decision**: Extend the existing API application factory after prediction-row
matching with one public-asset preparation operation. It publishes the matched
subset before application construction completes and returns rows enriched with
stable `video_url` and `thumbnail_url` values.

**Rationale**: Selection and media response composition are API serving
concerns. Startup already materializes prediction artifacts and thumbnails, so
this replaces that path without creating a second orchestration surface.

**Alternatives considered**:

- A Hydra or training script was rejected because training does not own which
  prediction subset is currently served and a separate command could be skipped.
- Request-time publication was rejected because responses could be slow or
  expose partially prepared state.
- Keeping both presigned and public modes was rejected because the repository
  explicitly removes superseded behavior.

## Storage Configuration And Authentication

**Decision**: Keep `AWS_ENDPOINT_URL`, `AWS_DEFAULT_REGION`, credentials, and
private `AWS_BUCKET`; add the required API setting
`AITRAF_PUBLIC_ASSET_BUCKET`. One authenticated storage client accesses both
buckets.

**Rationale**: The public bucket is on the same S3-compatible server and uses
the same authenticated write access. Its distinct permission is anonymous
download access, which is verified externally rather than encoded in API
credentials.

**Alternatives considered**:

- A separate public base URL was rejected because the endpoint is shared and
  the public URL is deterministically derived.
- Separate public-bucket credentials were rejected because the deployed account
  is expected to have access to both buckets; missing access must fail startup.

## Public Object Identity And URLs

**Decision**: Publish videos as `videos/<video_id>` and thumbnails as
`thumbnails/<video-stem>.jpg`. Require `video_id` to be a basename with a file
suffix, require its `s3_path` to use the configured private source bucket, and
construct a path-style public URL as
`<endpoint>/<public-bucket>/<percent-encoded-key>`.

**Rationale**: Dedicated public prefixes make the exposed surface clear and
keep video and thumbnail identities deterministic. URL construction needs no
mutable release manifest or additional configuration.

**Alternatives considered**:

- Preserving private source keys was rejected because source paths should not
  define the public layout and could expose unrelated hierarchy.
- A release prefix was rejected for this feature because the agreed behavior is
  additive stable assets by video identity; changed content at an existing
  identity is an explicit conflict.
- Using raw string concatenation was rejected because bucket and key segments
  require strict validation and URL encoding.

## Idempotence, Copying, And Conflicts

**Decision**: Inspect destination objects with object metadata requests. Missing
videos are copied server-side from the private bucket; missing thumbnails cause
one temporary source-video download, thumbnail generation, and upload. Published
objects carry source-provenance metadata. Existing objects are accepted only
when their provenance and observable source identity match; otherwise startup
fails without deliberately overwriting the object. Duplicate public identities
are resolved before storage calls and conflicting mappings fail.

**Rationale**: Server-side video copy avoids unnecessary transfer through the
API. Provenance distinguishes an idempotent restart from a key collision and
supports loud failure. A race is resolved by inspecting the destination after a
copy/upload conflict and accepting only the same source provenance.

**Alternatives considered**:

- Blindly trusting any existing key was rejected because it can serve unrelated
  media under a valid-looking URL.
- Unconditionally overwriting was rejected because retraining or concurrent
  startup could damage production assets.
- Downloading and re-uploading every video was rejected because the storage
  service can copy between buckets on the same endpoint.

## Presigning Removal Scope

**Decision**: Delete all API demo presigning behavior: the presigner protocol and
factory, expiration constant, application state, route/service callback,
signed-link fixtures and tests, and presigning documentation. Rows contain their
stable public URLs before request handling. Retain the generic core
`presign_s3_uri` helper because `aitraf-train` still uses it for offline
classification-miss reports, outside this serving feature.

**Rationale**: This provides one API behavior with no compatibility branch while
avoiding an unrelated change to a genuine training consumer.

**Alternatives considered**:

- Keeping the API presigner as fallback was rejected as prohibited legacy
  compatibility.
- Deleting the generic helper repository-wide was rejected because it would
  break an offline training feature not superseded by public demo assets.

## Validation Strategy

**Decision**: Combine focused API/core tests with a container/runtime smoke test.
The smoke test performs anonymous HTTP GETs for every returned asset, restarts
with identical inputs, and confirms stable queryless URLs. Source scanning proves
the API contains no presigning symbols or signed-link behavior.

**Rationale**: Mocked tests prove decisions and failures; only a real anonymous
GET proves the new bucket permission is actually public.

**Alternatives considered**:

- Authenticated storage checks alone were rejected because they cannot validate
  public download permissions.
- HTTP HEAD alone was rejected because a bucket policy may permit GET but not
  HEAD.
