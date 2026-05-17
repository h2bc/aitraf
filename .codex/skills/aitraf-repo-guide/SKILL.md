---
name: aitraf-repo-guide
description: Use when Codex needs to navigate or modify the AITRAF repository, understand project structure, locate training/data/evaluation code, add config parameters, choose commands, or follow repo conventions before making code changes.
---

# AITRAF Repo Guide

Use this skill before changing repo structure, training pipelines, configs, data ops, or model code.

## Core Layout

- `configs/`: Hydra configs. `configs/task/*.yaml` selects the task/data manifests. `configs/model/*.yaml` selects model-specific hyperparameters. Top-level configs such as `train.yaml`, `train_eval.yaml`, `eval.yaml`, `data_ops.yaml`, and `prepare.yaml` wire scripts and defaults together.
- `scripts/`: command entrypoints used by `Taskfile.yml`. Prefer running these through `task`, not direct ad hoc commands.
- `src/aitraf/data_ops/`: shared dataset/cache operations: label download, clip download, pose/bbox extraction, VideoMAE feature extraction.
- `src/aitraf/label_ops/`: pairwise label task creation/upload.
- `src/aitraf/tasks/<task>/prepare.py`: manifest/vocab creation for each task.
- `src/aitraf/tasks/<task>/<model>/training.py`: task-model training implementation.
- `src/aitraf/tasks/<task>/<model>/evaluation.py`: task-model evaluation implementation when present.
- `src/aitraf/processing/`: sample processing, transforms, label mapping helpers, model input construction.
- `src/aitraf/processing/models/`: model-specific processors and model wrappers used across tasks.
- `src/aitraf/models/`: lower-level model modules, especially non-Hugging Face models.
- `src/aitraf/metrics/`: reusable metric functions and plots.
- `src/aitraf/tracking/`: MLflow parameter mapping and tracking helpers.
- `.codex/skills/`: repo-local Codex skills. Use `metric-reporting` for experiment result tables and `aqa-experiment-plan` for AQA experiment planning.

## Commands

Use `task` commands from the repo root. Read `Taskfile.yml` for the available commands and their exact script entrypoints.

Use Hydra overrides for experiments instead of editing config files unless changing the default behavior intentionally.

Examples:

```bash
task train_eval -- task=score_prediction_ordinal model=video_mae_temporal_fusion model.seed=43
task data_ops -- video_mae_feature_extraction.enabled=true
```

## Task And Model Pattern

Tasks are user-facing prediction problems:

- `trick_classification`
- `score_prediction`
- `score_prediction_binary`
- `score_prediction_ordinal`
- `score_prediction_pairwise`
- `score_prediction_rank`

Models are implementations under those tasks:

- `pose_tcn`
- `video_mae`
- `video_mae_temporal_fusion`

The usual path for training code is:

```text
src/aitraf/tasks/<task>/<model>/training.py
```

The matching config files are usually:

```text
configs/task/<task>.yaml
configs/model/<model>.yaml
```

The script dispatch layer is in:

```text
scripts/train.py
scripts/train_eval.py
scripts/eval.py
```