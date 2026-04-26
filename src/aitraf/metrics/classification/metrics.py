"""Classification metric definitions."""

from functools import partial

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

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

__all__ = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
]
