# Aggressive Inline Trick Recognition and Performance Feedback

TODO

## Prerequisities

- Nvidia GPU with drivers supporting CUDA 12.6 or higher
- Docker



## Installation / Setup

1. Start the `dev-container`
2. Copy `.env.example` as  `.env` and set environment variables
3. Now you can run notebooks & finetuning pipelines

## Code Quality

- `make lint` – run Ruff checks
- `make format` – apply Ruff formatting

## Pipelines

- `make data` – execute the Hydra data pipeline entrypoint
- `make train-video-mae` – execute Hydra finetuning pipeline
- `make eval-video-mae` – run evaluation for an existing model URI
- `make train-eval-video-mae` – run training then evaluation in one go

## Jupyter Notebook

- `make jupyter` – launch a notebook server inside the uv-managed env

## Project Resources

- **S3 storage**
  - https://storage.h2bcweb.com
  - bucket: aitraf
- **Label Studio**
  - https://label.h2bcweb.com/
- **ML flow**
  - https://mlops.h2bcweb.com/

  
  
