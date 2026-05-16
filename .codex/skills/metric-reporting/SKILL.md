---
name: metric-reporting
description: Use when reporting ML experiment results, model evaluation metrics, ablations, or train/validation/test comparisons for this repo. Always structure metrics with dummy baselines and model variants as rows, and train/val/test metrics as columns.
metadata:
  short-description: Report ML metrics consistently
---

# Metric Reporting

Use this skill whenever reporting, comparing, summarizing, or interpreting ML experiment metrics in this repo.

## Required Format

Always include a dummy baseline row when available.

First table: metrics. Use rows for:
- `dummy`
- current/best model
- previous model
- ablation variants
- filtered-data variants

Use columns for train/validation/test metrics. Pick the actual metrics from the task or experiment; do not hard-code a fixed metric set.

```text
variant                         train <metric>  val <metric>  test <metric>
dummy                           ...             ...           ...
model/version A                 ...             ...           ...
model/version B                 ...             ...           ...
```

If train or validation metrics are unavailable, still keep the available split columns and state which split metrics were not logged or not pulled.

Second table: split distribution. When labels or manifests are available, include a train/validation/test distribution table before the takeaway. Use the task's natural target buckets/classes, and include `n` plus counts or percentages.

```text
split       n      class/label 1   class/label 2   class/label 3
train       ...    ...             ...             ...
val         ...    ...             ...             ...
test        ...    ...             ...             ...
```

If the experiment uses filtered data or a changed evaluation set, report the distribution for the exact splits being compared. If distribution data was not pulled, say that explicitly instead of guessing.

## Reporting Rules

- Use the metrics that are meaningful for the task or already logged by the experiment.
- Keep metric direction explicit in the takeaway when it is not obvious, e.g. whether higher or lower is better.
- Include the run name or run id for the best/current model when useful.
- Do not report only model metrics without the dummy baseline if dummy is available.
- Include the train/validation/test target distribution table when the split data is available.
- When comparing filtered datasets, label the evaluation set clearly, e.g. `full train -> full test`, `full train -> noamb test`.
- Do not mix filtered-test and full-test conclusions without saying the test set changed.

## Short Takeaway

After the table, add one short conclusion:

```text
Takeaway: variant A wins on the full test set; variant B only looks better on the filtered test set.
```

Keep the takeaway direct and tied to the table. Avoid broad theory unless the user asks for interpretation.
