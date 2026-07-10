# Quickstart: Validate The Core / ML Core Split

Run commands from the repository root. Use a clean worktree or preserve any
unrelated local changes before implementation validation.

## Prerequisites

- A supported Python version (`>=3.10,<3.14`)
- The repository package manager and current lockfile
- Existing credentials, model reference, configuration, and fixture required by
  the chosen representative ML smoke workflow
- A baseline output for that workflow captured from the pre-refactor revision

See [package-boundaries.md](./contracts/package-boundaries.md) for allowed imports
and [data-model.md](./data-model.md) for the classification and validation record.

## 1. Verify source ownership

```bash
test -d packages/aitraf-core/src/aitraf_core
test -d packages/aitraf-ml-core/src/aitraf_ml_core
test -f packages/aitraf-core/src/aitraf_core/storage/s3.py
test -f packages/aitraf-train/src/aitraf_train/storage/clips.py
```

Search the repository for old runtime imports. The result must contain no source
or test consumer; historical feature documentation may still describe the old
state.

```bash
rg -n 'from aitraf_core\.(inference|loading|pre_processing|processing)|import aitraf_core\.(inference|loading|pre_processing|processing)|aitraf_core\.storage\.clips' packages --glob '*.py'
```

Expected: no matches.

## 2. Validate lightweight core in isolation

Create a clean environment using the repository's standard package tooling and
install only `packages/aitraf-core`. Record its resolved dependency report.

Expected outcomes:

- installation succeeds;
- `aitraf_core`, `aitraf_core.cache`, and `aitraf_core.utils` import;
- the resolved environment contains Boto3 but no ML, video, or tracking runtime;
- invalid cache/JSON inputs fail with the documented explicit exception.

Run the core tests:

```bash
uv run pytest packages/aitraf-core/tests
```

## 3. Validate ML core

```bash
uv run pytest packages/aitraf-ml-core/tests
```

Expected: model-loading and any migrated runtime tests pass using
`aitraf_ml_core`. A validation case with a missing required artifact or invalid
runtime requirement must fail explicitly.

Confirm removed paths are unavailable in a clean environment containing the new
packages:

```bash
uv run python -c "import importlib.util; assert importlib.util.find_spec('aitraf_core.loading') is None"
uv run python -c "import importlib.util; assert importlib.util.find_spec('aitraf_core.processing') is None"
uv run python -c "import importlib.util; assert importlib.util.find_spec('aitraf_core.storage.clips') is None"
```

## 4. Validate train integration

```bash
uv run pytest packages/aitraf-train/tests
uv run python -m compileall -q packages/aitraf-core/src packages/aitraf-ml-core/src packages/aitraf-train/src packages/aitraf-api/src packages/aitraf-train/scripts
uv run python -c "import aitraf_core, aitraf_ml_core, aitraf_train, aitraf_api"
```

Expected: all commands pass, storage tests use `aitraf_train.storage`, and train
ML consumers import `aitraf_ml_core`.

## 5. Compare representative workflow behavior

Run the representative workflow selected during implementation with the same
versioned fixture, configuration, model reference, and seed used for the
pre-refactor baseline. Record the exact command in the validation record.

Expected outcomes:

- the command completes through the new package surfaces;
- the output retains its existing schema;
- deterministic fields are exactly equal;
- numerical fields meet the predeclared deterministic tolerance, if exact
  equality is not supported by the existing workflow.

Do not accept a fallback model, alternate artifact, repaired configuration, or
different input when comparing results.

## 6. Inspect dependency direction

Inspect the resolved workspace dependency tree and package manifests.

Expected relationships:

```text
aitraf-ml-core -> aitraf-core
aitraf-train -> aitraf-ml-core
aitraf-train -> aitraf-core (only for direct generic helper use)
```

`aitraf-core` must not point to ML core, train, model, tensor, tracking, or video
dependencies; Boto3 is allowed for shared S3 behavior. `aitraf-ml-core` must not
point to train or API.
