"""Hydra-managed task dataset preparation entrypoint."""

from __future__ import annotations

from typing import Callable

from dotenv import load_dotenv
from hydra import main
from omegaconf import DictConfig

from aitraf.logging import heading, setup_logging
from aitraf.tasks.score_prediction.prepare import (
    run_prepare as run_score_prediction_prepare,
)
from aitraf.tasks.score_prediction_binary.prepare import (
    run_prepare as run_score_prediction_binary_prepare,
)
from aitraf.tasks.score_prediction_rank.prepare import (
    run_prepare as run_score_prediction_rank_prepare,
)
from aitraf.tasks.trick_classifier.prepare import (
    run_prepare as run_trick_classification_prepare,
)


PrepareTarget = Callable[[DictConfig, DictConfig], None]


PREPARE_TARGETS: dict[str, PrepareTarget] = {
    "trick_classification": run_trick_classification_prepare,
    "score_prediction": run_score_prediction_prepare,
    "score_prediction_binary": run_score_prediction_binary_prepare,
    "score_prediction_rank": run_score_prediction_rank_prepare,
}


@main(config_path="../configs", config_name="prepare", version_base=None)
def run(cfg: DictConfig) -> None:
    load_dotenv()
    setup_logging()

    prepare_fn = PREPARE_TARGETS.get(str(cfg.task.name))
    if prepare_fn is None:
        available = ", ".join(sorted(PREPARE_TARGETS))
        raise RuntimeError(
            f"No prepare target registered for task='{cfg.task.name}'. "
            f"Available tasks: {available or 'none'}."
        )

    heading(f"Prepare Task: {cfg.task.name}")
    prepare_fn(cfg.task, cfg)


if __name__ == "__main__":
    run()
