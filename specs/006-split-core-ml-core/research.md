# Research: Split Core And ML Core

## Decision 1: Use a separate distribution and import namespace

**Decision**: Create distribution `aitraf-ml-core` with import namespace
`aitraf_ml_core`. Keep `aitraf-core` with namespace `aitraf_core`.

**Rationale**: Separate distributions are the only direct way for consumers to
install generic helpers without resolving ML dependencies. A distinct namespace
makes package ownership visible in imports and makes reverse-dependency checks
straightforward.

**Alternatives considered**:

- Optional `aitraf-core[ml]` extras: rejected because extras affect installation
  but do not prevent base modules from importing unavailable ML libraries.
- One distribution with lazy imports: rejected because it hides an architectural
  boundary behind runtime import behavior.
- Put all ML code in `aitraf-train`: rejected for this feature because the chosen
  product boundary explicitly preserves reusable ML runtime behavior separately
  from training/evaluation orchestration.

## Decision 2: Keep lightweight core free of ML dependencies

**Decision**: Retain cache, strict JSON/JSONL readers, and shared S3 primitives in
`aitraf-core`; Boto3 is its only third-party runtime dependency.

**Rationale**: These helpers are generic, typed, and useful without model or
model/video concerns. A manifest limited to Boto3 provides an unambiguous and
easily testable meaning of “lightweight with shared S3 support.”

**Alternatives considered**:

- Permit NumPy or Boto3 in core: rejected because neither is required by retained
  generic helpers and each expands the dependency footprint.
- Localize all helpers into train: rejected because generic cache and structured
  file contracts are intentionally the lightweight reusable surface selected by
  the feature.

## Decision 3: Move reusable ML behavior as one coherent tree

**Decision**: Move current inference, loading, preprocessing, processing, model
sampling/video behavior, and Hugging Face naming into `aitraf_ml_core`.

**Rationale**: These modules expose tensor/model/video types or exist solely to
support those capabilities. Moving the coherent tree avoids split namespaces and
conditional imports. ML core may import lightweight cache helpers from core.

**Alternatives considered**:

- Keep “pure-looking” model path or sampling helpers in core: rejected because
  their only consumers and concepts are ML-specific; dependency-free code is not
  automatically cross-feature core code.
- Split loaders, video, and inference into several new packages: rejected as
  premature package proliferation without independent consumers.

## Decision 4: Share S3 primitives and localize clip orchestration

**Decision**: Keep S3 settings, client, URI, existence, and presigning helpers in
`aitraf_core.storage.s3`; move clip downloads to `aitraf_train.storage.clips`;
and migrate API thumbnail/media behavior to the core helpers.

**Rationale**: API thumbnail/media behavior and train are two real consumers of
the S3 primitives. Clip downloading remains train-specific.

**Alternatives considered**:

- Move all storage to train: rejected because it would duplicate S3 behavior in
  API thumbnail generation.
- Create `aitraf-storage`: rejected because a new package is unjustified until a
  second product surface needs the same storage contract.
- Move storage to ML core: rejected because S3 object access is not inherently ML
  runtime behavior.

## Decision 5: Enforce boundaries through manifests and source validation

**Decision**: Test the core manifest for zero runtime dependencies, install core
in isolation, import all public modules, and statically reject imports of
`aitraf_ml_core` and prohibited heavy top-level modules anywhere under
`packages/aitraf-core/src`.

**Rationale**: Manifest checks catch declared dependencies; source/import checks
catch undeclared and transitive import coupling. Clean installation proves the
consumer experience.

**Alternatives considered**:

- Rely on review alone: rejected because the regression is mechanically
  detectable.
- Validate only `import aitraf_core`: rejected because package-level imports may
  pass while a public submodule remains coupled.
- Add a new architecture framework: rejected because a small focused test is
  sufficient for the current repository.

## Decision 6: Direct migration with behavioral equivalence

**Decision**: Rewrite all repository imports and delete old source paths in one
change. Compare one representative ML workflow output before and after using the
same input, configuration, model reference, and seed where applicable.

**Rationale**: The constitution prohibits compatibility scaffolding. The feature
changes ownership rather than algorithms, so output equivalence is the correct
behavioral check.

**Alternatives considered**:

- Temporary forwarding packages: rejected as legacy compatibility.
- Dual imports with `try/except`: rejected because they hide incomplete migration
  and make installed behavior environment-dependent.
- Metric-only comparison: rejected because unchanged deterministic inputs should
  retain the existing output contract directly.
