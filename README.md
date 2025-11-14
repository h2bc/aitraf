# Deep Learning-Based Aggressive Inline Trick Recognition and Performance Feedback

We are building a trick recognizer for aggressive inline skating. Right now the plan is to fine-tune VideoMAE (starting with `OpenGVLab/VideoMAEv2-Base`) on the new dataset we recorded and labeled ourselves.

`make data` kicks off the Hydra pipeline that:
1. downloads the latest labels from Label Studio,
2. checks the schema and writes train/val/test manifests, and
3. pulls the raw videos from S3, then runs the same 1:1 crop + clip slicing we use for VideoMAE training so finetuning jobs get clean inputs.

The rest is still up in the air. If finetuning falls short we may try GenAI-style models, pure pose inputs, cropped-person classifiers, or mixes of video + metadata. We’ll update this file as we learn more.

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

## Project Resources

- **S3 storage**
  - https://storage.h2bcweb.com
  - bucket: aitraf
- **Label Studio**
  - https://label.h2bcweb.com/
