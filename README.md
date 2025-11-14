# Deep Learning-Based Aggressive Inline Trick Recognition and Performance Feedback

TODO


## Prerequisites

- `uv` on your PATH
- `make` for the convenience targets (optional)

## Installation / Setup

1. `uv sync`
2. Run project commands with `uv run <cmd>` (e.g. `uv run pytest`, `uv run bash`)

## Code Quality

- `make lint` – run Ruff checks
- `make format` – apply Ruff formatting

## Pipelines

- `make data` – execute the Hydra data pipeline entrypoint

## Jupyter Notebook

- `make jupyter` – launch a notebook server inside the uv-managed env

## Git LFS

We use lfs storage for storing `data` and oter large files directories.

1. Install Git LFS and run the one-time setup:
	```bash
	sudo apt install git-lfs   # or brew install git-lfs
	git lfs install
	```
2. After cloning or pulling, fetch the large files:
	```bash
	git lfs pull
	```
3. When committing new heavy artifacts, track them before the first commit:
	```bash
	git lfs track "models/*.pt"
	git add .gitattributes models/my_new_model.pt
	```
   Use `git lfs ls-files` to verify what is tracked.

## Project Resources

- **S3 storage**
  - https://storage.h2bcweb.com
  - bucket: aitraf
- **Label Studio**
  - https://label.h2bcweb.com/
