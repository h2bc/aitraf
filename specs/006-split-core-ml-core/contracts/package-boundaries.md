# Package Boundary Contract

## `aitraf-core`

Distribution: `aitraf-core`  
Import namespace: `aitraf_core`

Public ownership is limited to:

- generic file-cache control in `aitraf_core.cache`;
- strict JSON and JSONL object readers in `aitraf_core.utils`;
- shared S3 client, URI, existence, and presigning helpers in `aitraf_core.storage.s3`.

The distribution declares Boto3 as its only runtime dependency. Public inputs and outputs
remain the single typed forms already documented by those functions. Missing
files, directories supplied as files, malformed JSON, and non-object records
raise explicit exceptions.

The following import categories are prohibited from all core source modules:

- `aitraf_ml_core` and `aitraf_train`;
- tensor/model libraries;
- video decoding/processing libraries;
- experiment tracking/model registry libraries;
- cloud SDKs other than Boto3 for the shared S3 surface.

## `aitraf-ml-core`

Distribution: `aitraf-ml-core`  
Import namespace: `aitraf_ml_core`

Public ownership includes reusable:

- classification and model inference;
- model loading from supported model/artifact systems;
- model preprocessing and feature cache path construction;
- frame sampling, video decoding, and tensor processing;
- model-specific processing for the existing Pose TCN and VideoMAE families;
- model-hub cache naming.

ML core may import `aitraf_core` for generic helpers. It must not import
`aitraf_train` or `aitraf_api`. Required dependencies and artifacts fail
explicitly when absent. Conditional imports of old `aitraf_core` ML paths are
not part of the contract.

## `aitraf-train`

Training retains offline data preparation, clip-download orchestration, model training,
evaluation, metrics, experiment tracking orchestration, configuration, and
scripts. Clip imports use `aitraf_train.storage`; shared S3 imports use
`aitraf_core.storage.s3`; ML runtime imports use `aitraf_ml_core`.

## Removed contracts

After migration, the following are intentionally invalid:

- `aitraf_core.inference`
- `aitraf_core.loading`
- `aitraf_core.pre_processing`
- `aitraf_core.processing`
- `aitraf_core.storage.clips`
- `aitraf_core.utils.huggingface`

No alias, forwarding module, re-export, dual import, or deprecation period is
provided. All in-repository consumers move atomically.

## Validation contract

The boundary is accepted only when:

1. `aitraf-core` installs and every public core module imports in a clean
   supported-Python environment.
2. Its resolved dependency set and manifest contain no third-party runtime
   dependency.
3. Static validation finds no prohibited source import in core.
4. Every moved source import in the repository uses the new owner.
5. Importing each removed path fails.
6. ML-core tests and affected train tests pass.
7. A representative fixed-input ML workflow produces an equivalent output
   before and after the migration.
