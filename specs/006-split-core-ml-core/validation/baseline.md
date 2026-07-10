# Pre-Migration Baseline

The deterministic local baseline used the existing core storage/model-loading
tests and the in-progress API thumbnail tests:

```bash
uv run pytest packages/aitraf-core/tests packages/aitraf-api/tests/features/demo_predictions/test_thumbnails.py -q
```

Result before source movement: `18 passed in 18.69s`.

No external model workflow was selected because the repository environment does
not provide a versioned local model artifact and fixture that can run without
external credentials. The migrated loader, storage, thumbnail, and complete
package tests provide deterministic contract equivalence for this package-only
refactor.
