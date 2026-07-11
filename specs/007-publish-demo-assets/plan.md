# Implementation Plan: Publish Demo Assets

**Branch**: `[007-publish-demo-assets]` | **Date**: 2026-07-11 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/007-publish-demo-assets/spec.md`

## Summary

Replace API demo-media presigning with an API-owned startup publication flow.
After classification and AQA prediction rows are matched, validate and
deduplicate the selected videos, copy missing videos from the private
`AWS_BUCKET` to deterministic `videos/` keys in the required
`AITRAF_PUBLIC_ASSET_BUCKET`, generate and upload only missing thumbnails under
`thumbnails/`, validate provenance for existing objects, and enrich the in-memory
rows with constant path-style public URLs derived from `AWS_ENDPOINT_URL`.
Delete every API presigner, expiration, callback, state, test, and documentation
path; retain the generic core presigner only for its unrelated offline training
consumer.

## Technical Context

**Language/Version**: Python `>=3.10,<3.14`

**Primary Dependencies**: FastAPI, Pydantic, python-dotenv, Boto3 through
`aitraf-core`, MLflow for existing prediction artifact loading, and FFmpeg for
thumbnail extraction

**Storage**: One S3-compatible endpoint and authenticated client; private source
bucket from `AWS_BUCKET`; publicly downloadable destination from required
`AITRAF_PUBLIC_ASSET_BUCKET`; deterministic `videos/<video_id>` and
`thumbnails/<video-stem>.jpg` objects with provenance metadata

**Testing**: Pytest API/core unit and startup integration tests, static API
presigning-removal scan, command-level API smoke test, and anonymous HTTP GET
validation of returned media URLs

**Target Platform**: Linux API container/runtime with network access to the
existing S3-compatible endpoint and FFmpeg installed

**Project Type**: Python monorepo with independently installable API, core, ML
runtime, and training packages

**Performance Goals**: Route requests perform zero storage operations and return
already-prepared URLs; identical second startup performs zero copy/upload work;
videos whose public copy already exists are not downloaded; only videos with
missing thumbnails require temporary local download

**Constraints**: The source and public buckets share endpoint and credentials;
the public bucket is distinct and grants anonymous download; startup fails
before readiness on missing config/assets, denied operations, invalid rows,
generation failures, or provenance conflicts; no overwrite, deletion, signed
URL fallback, release cleanup, or retraining orchestration is introduced

**Scale/Scope**: The matched precomputed demo subset only; API config, startup,
demo asset preparation, response construction, tests and docs; one generic core
server-side copy helper if needed; no training/evaluation behavior changes

## Constitution Check

*GATE: Passed before Phase 0 and re-checked after Phase 1 design.*

- **No Excessive Fallbacks**: PASS. Required public bucket configuration,
  storage access, row shapes, source objects, provenance, and generated
  thumbnails all fail explicitly. Private or presigned links are never returned
  as fallback.
- **Package By Feature**: PASS. Demo selection, publication orchestration,
  thumbnail generation, URL preparation, and response behavior remain under
  `packages/aitraf-api`. Only a dependency-light server-side S3 copy primitive
  used at the storage boundary is eligible for `aitraf-core`; train is not given
  a parallel publisher.
- **Function Decomposition**: PASS. Design separates row validation and
  derivation, public URL construction, destination inspection, video copy,
  thumbnail materialization, provenance validation, and row enrichment.
- **Functional Programming And State**: PASS. Asset identities and URLs are
  immutable derived values. Network/filesystem state stays inside startup
  storage and thumbnail boundaries; requests consume prepared rows only.
- **Reproducibility**: PASS. Required run IDs, source/public bucket settings,
  endpoint, credentials, tests, startup command, anonymous GET checks, and
  restart validation are documented in the quickstart.
- **No Legacy Compatibility**: PASS. API presigner protocols, factories,
  expiration constants, application state, callback parameters, tests, fixtures,
  and documentation are deleted. There is no dual serving mode. The generic
  core helper remains solely for an existing offline train consumer and is not
  an API compatibility path.
- **Required Types Over Defensive Normalization**: PASS. The API accepts one
  required prediction media shape and one explicit public bucket setting.
  Invalid IDs, cross-bucket URIs, alternate thumbnail paths, and conflicting
  public mappings are rejected.

## Project Structure

### Documentation (this feature)

```text
specs/007-publish-demo-assets/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── public-demo-assets.md
└── tasks.md              # generated later by /speckit-tasks
```

### Source Code (repository root)

```text
packages/
├── aitraf-api/
│   ├── README.md
│   ├── src/aitraf_api/
│   │   ├── app.py
│   │   ├── config.py
│   │   └── features/demo_predictions/
│   │       ├── assets.py          # selected asset derivation/publication
│   │       ├── route.py           # prepared-row access only
│   │       ├── schemas.py
│   │       ├── service.py         # direct public URL response mapping
│   │       └── thumbnails.py      # thumbnail extraction boundary
│   └── tests/
│       ├── test_config.py
│       └── features/demo_predictions/
│           ├── test_assets.py
│           ├── test_demo_predictions.py
│           └── test_thumbnails.py
├── aitraf-core/
│   ├── src/aitraf_core/storage/s3.py
│   └── tests/test_storage_s3.py
└── aitraf-train/                  # unchanged

.env.example
AGENTS.md
```

**Structure Decision**: Replace the mixed API `videos.py` download/presign
surface with `assets.py`, which owns strict selected-asset derivation,
deterministic public identity/URL construction, destination inspection,
provenance checks, and startup publication. Keep FFmpeg extraction in
`thumbnails.py` but remove its private-bucket row mutation and make it a focused
generation boundary invoked only for missing public thumbnails. Add only the
generic S3 copy/inspection primitive needed at the storage boundary to core.
The route and service consume prepared `video_url` and `thumbnail_url` fields
directly and hold no client or callback state.

## Phase 0: Research

See [research.md](./research.md). Decisions cover API ownership, shared endpoint
configuration, deterministic public layout, server-side video copies,
provenance-based idempotence and conflicts, API presigning removal scope, and
validation against real anonymous access.

## Phase 1: Design And Contracts

See [data-model.md](./data-model.md),
[contracts/public-demo-assets.md](./contracts/public-demo-assets.md), and
[quickstart.md](./quickstart.md).

## Implementation Direction

1. Extend strict API settings and `.env.example` with required
   `AITRAF_PUBLIC_ASSET_BUCKET`; validate that endpoint and both bucket names are
   usable and that source and public buckets differ.
2. Add focused shared storage primitives for destination metadata inspection
   and server-side object copy while preserving existing core consumers.
3. Introduce immutable API selected/public asset representations plus pure
   helpers for row validation, key derivation, URL encoding, deduplication, and
   prepared-row enrichment.
4. Replace the current startup thumbnail-first flow with one matched-subset
   preparation pass: inspect/copy videos, inspect/generate/upload thumbnails,
   validate provenance, then attach stable URLs.
5. Remove API video downloads except the temporary downloads required for
   missing thumbnails; never download already-complete public assets.
6. Change route/service composition to read stable URL fields directly from
   prepared application rows and perform no request-time storage operation.
7. Delete the complete API presigning implementation and references, including
   expiration constants, protocols, factories, app state, callback arguments,
   signed-link tests/fixtures, and README language. Do not add aliases, feature
   flags, or fallback behavior.
8. Replace existing video/thumbnail tests with publication, provenance,
   subset-only, idempotence, failure, and stable-response tests; retain the
   generic core presigning helper/tests required by offline training.
9. Run package tests and the zero-presigning source scan, then validate the real
   configured bucket using startup, repeated startup, and anonymous HTTP GETs
   for every returned video and thumbnail.

## Post-Design Constitution Check

- **No Excessive Fallbacks**: PASS. The contract defines explicit startup
  failures and prohibits returning any alternate media location.
- **Package By Feature**: PASS. API orchestration stays with demo predictions;
  core additions are storage primitives only; train is unchanged.
- **Function Decomposition**: PASS. Data derivation and side-effecting storage
  operations have distinct contracts and test boundaries.
- **Functional Programming And State**: PASS. Derived asset values are immutable
  and application state contains completed rows, not mutable presigner/client
  callbacks.
- **Reproducibility**: PASS. Design artifacts specify all inputs and executable
  validation, including anonymous public access and restart behavior.
- **No Legacy Compatibility**: PASS. The external API response field names stay
  current, but their implementation moves directly from expiring to public URLs;
  all old API machinery is removed.
- **Required Types Over Defensive Normalization**: PASS. Contracts define one
  media row, configuration, selected asset, provenance, and prepared-row shape.

## Complexity Tracking

No constitution violations require justification.
