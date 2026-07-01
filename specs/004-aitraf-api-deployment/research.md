# Research: AITRAF API Deployment

## Decision: Add an API-specific Dockerfile under `packages/aitraf-api`

**Rationale**: The API deployment artifact is owned by the serving package. The
existing train Dockerfile already establishes the repository's image pattern:
root build context, CUDA runtime base, `uv` copied from the official image,
frozen workspace sync, and package-scoped install. The API Dockerfile can reuse
that pattern while changing the package target to `aitraf-api`.

**Alternatives considered**:

- Reuse `packages/aitraf-train/Dockerfile` for API serving. Rejected because it
  installs train dependencies and starts with `sleep infinity`, which is not a
  production API entrypoint.
- Add a root Dockerfile with build arguments for train/API. Rejected for this
  feature because the user asked for a new API Dockerfile and package ownership
  is clearer under `packages/aitraf-api`.

## Decision: Install only API and Core dependencies in the API image

**Rationale**: `packages/aitraf-api/pyproject.toml` depends on `aitraf-core`, so
`uv sync --frozen --no-dev --no-editable --package aitraf-api` should install
the API package and the core runtime dependency without pulling in
`aitraf-train`. Copying only `packages/aitraf-api` and `packages/aitraf-core`
into the build context path used by the Dockerfile reinforces that train code is
not part of the API image.

**Alternatives considered**:

- Install the root workspace package. Rejected because root dependencies include
  `aitraf-train`.
- Install all workspace packages then rely on runtime discipline. Rejected
  because it bloats the serving image and violates the requested dependency
  boundary.

## Decision: Copy `data/` and filtered demo clips into the API image but keep full `storage/` external

**Rationale**: Local inspection shows `data/` is small and contains manifests and
vocabularies needed by API demo selection, while full `storage/` is very large
and contains runtime artifacts such as all clips, feature caches, and model
cache/state. The API image should include the small versioned data inputs and
only the demo clips selected by the API demo-video filter. `AITRAF_STORAGE_PATH`
still points at image-bundled or mounted runtime storage.

**Alternatives considered**:

- Copy both `data/` and full `storage/` into the image. Rejected because
  `storage/` is large runtime state and would make builds slow, fragile, and
  unreproducible.
- Copy neither `data/` nor `storage/`. Rejected because manifests/vocabularies in
  `data/` are small repo-owned inputs and useful for reproducible demo behavior.
- Create a separate committed `data/demo-clips` directory. Rejected because it
  duplicates clips already produced under `storage/data/clips`.

## Decision: Extend the existing image workflow instead of adding a new workflow

**Rationale**: The repository already publishes `aitraf-train` to GHCR on
`master` and manual dispatch with `latest` and short-SHA tags. Extending that
workflow keeps registry, permissions, cache, and metadata behavior consistent
while adding the new `ghcr.io/<owner>/aitraf-api` image location.

**Alternatives considered**:

- Add a separate `publish-api-image.yml`. Rejected because the user explicitly
  asked to extend the existing workflow and because shared conventions are
  easier to maintain in one Docker-image publishing workflow.
- Replace train publishing with a matrix that treats both images identically.
  Rejected for the initial implementation because API publishing has an
  additional test gate that train publishing does not currently have.

## Decision: Keep train publishing independent from API test/publish

**Rationale**: The user requested API tests before deploying and specified that
API test failure should kill the API branch while train can continue. The
workflow should therefore use independent train and API jobs, with API publish
depending on API tests and train publish having no dependency on the API test
job.

**Alternatives considered**:

- Put train and API publishing in one sequential job. Rejected because an API
  test failure would block train publishing.
- Use one matrix job for both images. Rejected because the API image has a
  distinct pre-publish test requirement.

## Decision: Publish API image as `ghcr.io/<owner>/aitraf-api`

**Rationale**: This mirrors the existing train image naming convention under the
same GHCR owner while giving API deployments a distinct image reference.

**Alternatives considered**:

- Publish under the train image with a tag suffix. Rejected because API and
  train images are different deployment artifacts.
- Publish to a different registry. Rejected because the existing repository
  convention already uses GHCR.
