# aitraf-core

`aitraf-core` contains lightweight behavior shared by the API, ML runtime, and
training surfaces without pulling in the ML dependency stack.

Public surfaces:

- `aitraf_core.cache`: generic file-cache control.
- `aitraf_core.utils`: strict JSON and JSONL object readers.
- `aitraf_core.storage.s3`: shared S3 settings, client creation, URI parsing,
  object existence, key iteration, and URL presigning.

Boto3 is the package's only third-party runtime dependency. Model loading,
inference, tensors, video decoding, and model preprocessing belong to
`aitraf-ml-core`. Clip-download orchestration belongs to `aitraf-train`.

Missing settings, invalid URIs, missing files, and invalid structured-file shapes
raise explicit errors; callers must not rely on fallback input representations.

```bash
uv run pytest packages/aitraf-core/tests
```
