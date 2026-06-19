# Research: API Inference Surface

## Decision: Use FastAPI in `packages/aitraf-api`

**Rationale**: The user explicitly selected FastAPI. It matches the existing Python
workspace, supports typed request/response schemas, dependency-injected auth and
services, and straightforward pytest coverage with a test client.

**Alternatives considered**: Flask was rejected because the user chose FastAPI and
the typed schema/dependency model is a better fit. A CLI-only serving path was
rejected because the Next.js frontend needs HTTP endpoints.

## Decision: Keep endpoints parameter-minimal

**Rationale**: Health and demo-video listing take no parameters. Each inference
endpoint accepts only the selected video id. This keeps the demo surface bounded
to current test-set examples and avoids exposing arbitrary video uploads or
model configuration through public request bodies.

**Alternatives considered**: Accepting video ids plus model options was rejected
because it widens the API and makes reproducibility weaker. Accepting uploaded
videos was rejected as out of scope.

## Decision: Use matching current test-manifest videos as the demo source of truth

**Rationale**: `data/manifests/trick_classification/test.jsonl` and
`data/manifests/score_prediction_ordinal/test.jsonl` already represent the
current bounded test sets. The demo-video endpoint should return a flat display
list of matching videos and metadata, keyed by `video_id`, without exposing task
grouping. Each inference endpoint then resolves the selected id against its own
current test manifest.

**Alternatives considered**: Creating a new demo manifest was rejected because it
duplicates current test-set state. Querying labels or training outputs directly
was rejected because the test manifests already contain the display and selection
data needed for the demo.

## Decision: Map trick AQA to `score_prediction_ordinal`

**Rationale**: The repo's current quality-assessment task surface is ordinal score
prediction with `execution_score` labels. This matches the requested trick AQA
output better than regression or pairwise/rank variants for a first API.

**Alternatives considered**: `score_prediction` regression was rejected for this
API because the current spec and previous repo plan emphasize ordinal score
workflows as the validated quality-assessment surface. Pairwise/rank tasks were
rejected because their inputs are not a single selected video id.

## Decision: Use explicit model URIs plus shared core inference

**Rationale**: The API should not duplicate training/evaluation inference logic
or own model preprocessing constants. The serving surface receives a selected
test-set `video_id`, resolves the row, loads the registered MLflow model URI
configured for that endpoint, and delegates video processing, processor use,
prediction, and decoding to `aitraf-core`.

**Final split**:

- `.env`/environment defines the required data root, storage root, MLflow model
  URIs, and tracking URI.
- API config defines endpoint wiring, compact demo-facing model kind, manifest
  path suffixes, and the clips path suffix.
- MLflow/core provide the trained model artifact, image processor, frame count,
  sampling strategy, and label decoding path used by inference.

**Alternatives considered**: Reading whatever latest checkpoint exists in
`storage/runs` was rejected because it is ambiguous and unreproducible.
Hardcoded implicit preprocessing defaults were rejected because they hide
model/preprocessing drift. Importing `aitraf-train` evaluation code from the API
was rejected because train remains the offline workflow package.

## Decision: Inference pipeline loads videos and runs the model path, not precomputed dummy outputs

**Rationale**: The API is intended to demonstrate actual request-time inference:
resolve manifest row, load video, sample frames, run the model's image processor,
and predict with the configured model. This exercises the serving path that the
frontend will use.

**Alternatives considered**: Returning labels from manifests, cached evaluation
predictions, or dummy predictions was rejected because it would not implement the
requested inference surface and would hide missing model artifacts.

## Decision: Use focused API tests with Arrange/Act/Assert structure and stubbed predictors

**Rationale**: Tests should prove endpoint contracts, auth, manifest selection,
and route orchestration without spending time or compute on model quality. A
stubbed predictor keeps tests fast and deterministic while still verifying that
valid ids reach the inference service. Fixture names and direct assertions should
make the arrangement and expected behavior clear without step-by-step comments.

**Alternatives considered**: Loading real model weights in tests was rejected as
slow, brittle, and outside the requested scope. Testing model performance or
metrics was explicitly rejected by the user and remains out of scope.

## Decision: Token-based auth for protected endpoints

**Rationale**: The API is for a controlled Next.js frontend. A shared bearer token
or equivalent token header is sufficient for this internal demo surface and keeps
deployment simple. The token must come from environment/configuration, never from
committed files.

**Alternatives considered**: Full user auth/OAuth was rejected as unnecessary for
this app-to-API integration. No auth was rejected by the spec.
