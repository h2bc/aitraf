# Research: Demo Clip Download

## Decision: Put generic S3 and clip download behavior in `aitraf-core`

**Rationale**: Both API runtime hydration and train data preparation need the
same low-level behavior: parse S3 clip URIs, load AWS-compatible settings, build
an object storage client, skip existing files unless forced, create destination
directories, and fail when downloads cannot complete. This behavior is reusable
runtime infrastructure and fits `aitraf-core` better than either feature-specific
package.

**Alternatives considered**:

- Keep S3 helpers in `aitraf-train` and import them from API. Rejected because it
  makes serving depend on train orchestration and violates package ownership.
- Duplicate S3 helpers in `aitraf-api`. Rejected because it creates divergent
  error handling and object-storage behavior.
- Keep API and train completely separate. Rejected because the feature goal is to
  avoid duplicated clip download behavior.

## Decision: Keep API-owned demo selection and hydration policy in `aitraf-api`

**Rationale**: The API owns the demo-video product surface: which manifests are
used, which selected IDs are shown to clients, and when runtime startup should
hydrate clips. Core should not know about API demo semantics, and train should
not control serving startup behavior.

**Alternatives considered**:

- Move demo selection into `aitraf-core`. Rejected because demo selection is API
  product behavior, not reusable model/runtime processing.
- Let train Hydra decide API demo clips. Rejected because startup serving behavior
  should not depend on offline pipeline orchestration.

## Decision: Runtime demo downloads are opt-in

**Rationale**: Downloading clips performs network and filesystem side effects.
The repository convention is to break loudly and avoid hidden defaults. An
explicit runtime setting makes deployments choose between mounted/prehydrated
storage and startup hydration.

**Alternatives considered**:

- Always download missing demo clips at startup. Rejected because it hides
  deployment state and makes startup unexpectedly depend on network credentials.
- Never download at startup. Rejected because servers without prehydrated storage
  cannot run the demo without manual copying.

## Decision: API image build does not bundle clips

**Rationale**: GitHub-hosted runners do not have repo-local `storage/data/clips`.
Publishing a clip-bundled image would require CI to download clips before build
or building from a machine with local storage. The current local-clip build
context should be removed. The feature goal is runtime availability, not
immutable clip bundling. Restoring image builds to be independent from clips
preserves reproducibility and avoids large image churn.

**Alternatives considered**:

- Copy filtered clips into the image during Docker build. Rejected because GHCR
  publishing cannot rely on local storage and would fail or require separate
  CI hydration.
- Commit demo clips under `data/`. Rejected because clips are generated/external
  artifacts and should not become normal source files.

## Decision: Train data ops adapts labels to shared core download requests

**Rationale**: The existing train data pipeline already owns label parsing,
Hydra config, progress logging, and force behavior. It should retain that
orchestration while delegating actual clip downloads to core. This minimizes
behavioral change while removing duplicated S3 mechanics.

**Alternatives considered**:

- Move the full train `download_clips` data op to core. Rejected because it would
  pull train-specific label schema, pandas behavior, and pipeline concerns into
  shared runtime code.
- Leave train unchanged and only add core for API. Rejected because shared code
  would not actually be reused by both surfaces.

### Current S3 Helper Import Targets

Local inspection found the following train call sites using the existing S3
helpers:

- `packages/aitraf-train/src/aitraf_train/preparation/data_ops/download_clips.py`
- `packages/aitraf-train/src/aitraf_train/preparation/data_ops/download_labels.py`
- `packages/aitraf-train/src/aitraf_train/preparation/data_ops/download_pairwise_labels.py`
- `packages/aitraf-train/src/aitraf_train/preparation/label_ops/upload_pairs.py`
- `packages/aitraf-train/src/aitraf_train/metrics/classification/plot.py`

These call sites now import shared S3 helpers from `aitraf_core.storage.s3`.
The obsolete train-owned `s3_utils.py` module was removed instead of kept as a
compatibility re-export.

## Decision: Fail the API startup hydration when any selected clip is unresolved

**Rationale**: The demo list and inference routes should not silently diverge.
If a selected demo row has no source, an invalid source, unavailable credentials,
or a failed download, startup hydration should fail instead of silently removing
the clip from the demo set or allowing later inference failures.

**Alternatives considered**:

- Skip failed clips and continue. Rejected because it hides data/deployment
  issues and changes the demo set implicitly.
- Defer all failures to inference time. Rejected because startup validation gives
  operators a clear readiness signal.
