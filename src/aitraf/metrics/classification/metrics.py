"""Classification metric definitions."""

from functools import partial

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from ..metrics import EvalMetric

accuracy = EvalMetric(
    name="accuracy",
    callback=accuracy_score,
)

balanced_accuracy = EvalMetric(
    name="balanced_accuracy",
    callback=balanced_accuracy_score,
)

f1_macro = EvalMetric(
    name="f1_macro",
    callback=partial(
        f1_score,
        average="macro",
        zero_division=0,
    ),
)

f1_binary = EvalMetric(
    name="f1",
    callback=partial(
        f1_score,
        zero_division=0,
    ),
)

precision = EvalMetric(
    name="precision",
    callback=partial(
        precision_score,
        zero_division=0,
    ),
)

recall = EvalMetric(
    name="recall",
    callback=partial(
        recall_score,
        zero_division=0,
    ),
)

__all__ = [
    "accuracy",
    "balanced_accuracy",
    "f1_binary",
    "f1_macro",
    "precision",
    "recall",
]
