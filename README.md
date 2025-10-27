# Deep Learning-Based Aggressive Inline Trick Recognition and Performance Feedback

TODO


## Prerequisites

- Ensure `uv` is installed and available on your PATH.

## Installation / Setup

1. **Sync project dependencies**
	```bash
	uv sync --group dev
	```
	This creates the local `.venv` managed by uv. Omit `--group dev` if you only need the runtime dependencies.

2. **Use the virtual environment**
	Activate it with `source .venv/bin/activate`, or prefix commands with `uv run`, e.g. `uv run pytest`.

3. **Jupyter Notebook**
	Launch Jupyter inside the environment:
	```bash
	uv run jupyter notebook
	```

## Code Quality

- **Ruff** (lint & format): `uv run ruff check .` to lint, `uv run ruff format .` to apply formatting.

## Project Resources

- **S3 storage**
  - https://storage.h2bcweb.com
  - bucket: aitraf
- **Label Studio**
  - https://label.h2bcweb.com/
