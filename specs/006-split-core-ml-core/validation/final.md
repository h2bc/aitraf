# Final Validation

Completed validation:

```bash
uv run ruff check packages/aitraf-core packages/aitraf-ml-core packages/aitraf-train packages/aitraf-api
uv run pytest packages/aitraf-core/tests packages/aitraf-ml-core/tests packages/aitraf-train/tests packages/aitraf-api/tests -q
uv run python -m compileall -q packages/aitraf-core/src packages/aitraf-ml-core/src packages/aitraf-train/src packages/aitraf-api/src packages/aitraf-train/scripts
uv run python -c "import aitraf_core, aitraf_ml_core, aitraf_train, aitraf_api"
```

Results: lint passed; 33 tests passed; compilation and all four top-level imports passed. Removed-path checks
confirm the former core ML and clip paths are unavailable.

The API container also builds successfully with the workspace core dependency:

```bash
docker build -f packages/aitraf-api/Dockerfile -t aitraf-api:core-split .
```
