# Data Model: Split Core And ML Core

This feature changes code ownership and dependency contracts rather than
application data. The following entities describe the architecture state that
implementation and validation must establish.

## Package Surface

Represents an independently installable repository package.

### Fields

- **distribution_name**: Required unique package name.
- **import_namespace**: Required unique Python import root.
- **responsibility**: Required single ownership statement.
- **runtime_dependencies**: Exact declared direct dependency set.
- **public_modules**: Modules documented for consumers.
- **consumers**: In-repository packages or workflows importing the surface.

### Instances

| Distribution | Namespace | Responsibility | Allowed dependency character |
|--------------|-----------|----------------|------------------------------|
| `aitraf-core` | `aitraf_core` | Generic cache, strict structured-file, and shared S3 helpers | Python standard library plus Boto3 |
| `aitraf-ml-core` | `aitraf_ml_core` | Reusable model, tensor, inference, preprocessing, sampling, and video runtime | May use ML/video libraries and `aitraf-core` |
| `aitraf-train` | `aitraf_train` | Offline data, storage, training, evaluation, metrics, tracking, configs, scripts | May use both core packages and training/storage libraries |
| `aitraf-api` | `aitraf_api` | Serving behavior using core S3 helpers for thumbnails and media | Depends on core, never ML core |

### Validation rules

- Distribution and namespace names are unique.
- Core has no declared third-party runtime dependency other than Boto3.
- Core source imports neither ML core nor a prohibited ML/video module.
- ML core may depend on core; core may not depend on ML core.
- Package initializers may export only owner-defined public symbols.

## Module Classification

Represents the required migration decision for one current core module or
coherent module tree.

### Fields

- **current_path**: Required existing source path.
- **classification**: Exactly one of `retain`, `move_ml_core`, `move_train`,
  `decompose`, or `remove`.
- **destination_path**: Required except when classification is `remove`.
- **dependency_category**: `stdlib`, `storage`, `model_hub`, `tracking`, `tensor`,
  `video`, or a documented combination.
- **consumer_set**: All current in-repository importers.
- **contract_change**: Required statement of preserved or deliberately updated
  public input/output contract.

### Validation rules

- Every current core source module appears exactly once.
- No two destinations provide duplicate implementations of one capability.
- A retained core module has only standard-library imports, except the declared S3 surface may import Boto3.
- Shared S3 primitives remain in core; clip orchestration with only train consumers moves to train.
- Any old path classified as moved or removed is absent after migration.

### Lifecycle

`inventoried → classified → moved/retained/removed → callers_updated → validated`

A module cannot be considered validated while its old path or a forwarding alias
remains.

## Dependency Boundary Rule

Represents an enforceable architectural invariant.

### Fields

- **subject_package**: Package being constrained.
- **allowed_dependencies**: Exact package/category allowlist.
- **prohibited_dependencies**: Exact package/category denylist where useful.
- **validation_method**: Manifest, source import, clean install, or runtime import
  check.
- **failure_evidence**: Required actionable validation error.

### Required relationships

```text
aitraf-core
    ↑
aitraf-ml-core
    ↑
aitraf-train

aitraf-train ─────→ aitraf-core
aitraf-api   (unchanged by this feature)
```

`aitraf-train` may depend on both core packages. The arrows prohibit reverse
imports from lightweight core into ML core and prohibit either core package from
depending on train or API.

## Validation Record

Represents reproducible evidence that the refactor preserves behavior.

### Fields

- **package_manifests**: Versions of each changed manifest and lockfile.
- **dependency_reports**: Resolved dependency set for isolated core and full ML
  environments.
- **commands**: Exact test, compile, import, and smoke commands.
- **workflow_fixture**: Versioned representative input.
- **workflow_configuration**: Exact configuration, model reference, and seed if
  applicable.
- **before_output**: Baseline output captured before migration.
- **after_output**: Output captured after migration.
- **comparison_result**: Exact equality or documented deterministic tolerance for
  numerical output.

### Validation rules

- Core dependency report contains none of the prohibited dependency categories.
- All package and boundary tests pass.
- Before and after outputs satisfy the same existing output contract.
- Missing artifact/dependency and invalid core input checks fail explicitly.
