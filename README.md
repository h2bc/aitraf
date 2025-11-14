# Deep Learning-Based Aggressive Inline Trick Recognition and Performance Feedback

TODO

## Prerequisites

- Linux environment with NVIDIA GPU
- `uv` on your PATH
- `make`


## Installation / Setup

1. `make install`
2. Run project commands with `uv run <cmd>` (e.g. `uv run pytest`, `uv run bash`)

## Code Quality

- `make lint` – run Ruff checks
- `make format` – apply Ruff formatting

## Pipelines

- `make data` – execute the Hydra data pipeline entrypoint

## Jupyter Notebook

- `make jupyter` – launch a notebook server inside the uv-managed env

## Project Resources

- **S3 storage**
  - https://storage.h2bcweb.com
  - bucket: aitraf
- **Label Studio**
  - https://label.h2bcweb.com/
