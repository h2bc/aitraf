# Core Isolation Validation

- Manifest direct runtime dependency: `boto3` only.
- Automated AST boundary scan rejects ML core, train, model, tensor, video, and
  tracking imports.
- Core cache, JSON/JSONL, and S3 tests pass.
- API and train consume `aitraf_core.storage.s3`; ML core consumes
  `aitraf_core.cache`.

Command:

```bash
uv run pytest packages/aitraf-core/tests -q
```
