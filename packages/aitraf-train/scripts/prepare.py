"""Hydra-managed task dataset preparation entrypoint."""

from __future__ import annotations

from typing import Callable

from dotenv import load_dotenv
from hydra import main
from omegaconf import DictConfig

from aitraf_train.logging import heading, setup_logging
from aitraf_train.tasks.score_prediction.prepare import (
    run_prepare as run_score_prediction_prepare,
)
from aitraf_train.tasks.score_prediction_binary.prepare import (
    run_prepare as run_score_prediction_binary_prepare,
)
from aitraf_train.tasks.score_prediction_pairwise.prepare import (
    run_prepare as run_score_prediction_pairwise_prepare,
)
from aitraf_train.tasks.score_prediction_ordinal.prepare import (
    run_prepare as run_score_prediction_ordinal_prepare,
)
from aitraf_train.tasks.trick_classifier.prepare import (
    run_prepare as run_trick_classification_prepare,
)


PrepareTarget = Callable[[DictConfig, DictConfig], None]


PREPARE_TARGETS: dict[str, PrepareTarget] = {
    "trick_classification": run_trick_classification_prepare,
    "score_prediction": run_score_prediction_prepare,
    "score_prediction_binary": run_score_prediction_binary_prepare,
    "score_prediction_ordinal": run_score_prediction_ordinal_prepare,
    "score_prediction_pairwise": run_score_prediction_pairwise_prepare,
}


@main(config_path="../configs", config_name="prepare", version_base=None)
def run(cfg: DictConfig) -> None:
    load_dotenv()
    setup_logging()

    prepare_cfg = cfg.create_manifests
    prepare_fn = PREPARE_TARGETS.get(str(cfg.task.name))
    if prepare_fn is None:
        available = ", ".join(sorted(PREPARE_TARGETS))
        raise RuntimeError(
            f"No prepare target registered for task='{cfg.task.name}'. "
            f"Available tasks: {available or 'none'}."
        )

    if prepare_cfg.enabled:
        heading(f"Create Manifests: {cfg.task.name}")
        prepare_fn(cfg.task, prepare_cfg)
    else:
        heading("Skip manifest creation (disabled)")


if __name__ == "__main__":
    run()
