# ML Runtime Migration Validation

The MLflow loader contract tests moved unchanged to the `aitraf_ml_core`
namespace. They verify CPU map-location behavior, device mapping, and explicit
rejection of an unsupported device. All train callers compile against the new
namespace, and the full train test suite passes.

The API thumbnail tests retain the baseline path and upload outcomes while using
core client, URI parsing, object existence, and presigning helpers.

```bash
uv run pytest packages/aitraf-ml-core/tests packages/aitraf-train/tests packages/aitraf-api/tests -q
```
