# aitraf-ml-core

`aitraf-ml-core` owns reusable heavy ML runtime behavior under the
`aitraf_ml_core` namespace:

- model loading from Hugging Face and MLflow;
- classification and model-specific inference;
- model preprocessing and feature extraction;
- tensor sampling and processing;
- video decoding;
- Pose TCN and VideoMAE runtime processing.

The package depends on `aitraf-core` for generic cache behavior. It never depends
on `aitraf-train` or `aitraf-api`. Training, evaluation, metrics, tracking, and
workflow orchestration remain in `aitraf-train`.

```bash
uv run pytest packages/aitraf-ml-core/tests
```
