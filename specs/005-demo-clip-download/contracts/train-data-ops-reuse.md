# Contract: Train Data Ops Reuse

## Purpose

Preserve existing train data pipeline behavior while reusing the shared core clip
download surface.

## Inputs

- Existing train labels JSONL input
- Existing output clip directory
- Existing force setting
- Existing object storage environment settings

## Required Behavior

- The train data-op continues to read labels and determine clip sources according
  to the existing data pipeline contract.
- It converts label-derived clip sources into shared core clip download requests.
- It stores clips in the same paths as before.
- It preserves existing force/skip semantics.
- It continues to report useful progress and summary information from the train
  pipeline surface.

## Prohibited Behavior

- Train data ops must not duplicate low-level S3 parsing/client/download logic
  once the shared core surface exists.
- Core must not import train label schema or Hydra config.

## Validation Expectations

- A representative labels input produces the same destination clip paths before
  and after the refactor.
- Existing train data-op command behavior remains compatible with current config
  usage.
