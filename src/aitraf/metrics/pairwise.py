"""Pairwise ranking metrics utilities."""

from typing import Callable, Sequence

from sklearn.metrics import accuracy_score


def build_pairwise_metrics() -> Callable[[Sequence[int], Sequence[int]], dict[str, float]]:
    def _compute_metrics(
        predictions: Sequence[int], labels: Sequence[int]
    ) -> dict[str, float]:
        return {
            "accuracy": accuracy_score(labels, predictions),
        }

    return _compute_metrics


__all__ = ["build_pairwise_metrics"]
