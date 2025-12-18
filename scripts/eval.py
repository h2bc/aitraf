"""Unified evaluation entrypoint dispatching per task/model combo."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from hydra import main
from omegaconf import DictConfig

from aitraf.logging import logger, setup_logging
from aitraf.tasks.trick_classifier.pose_tcn import (
    PoseTCNEvalConfig,
    run_evaluation as run_pose_tcn_evaluation,
)
from aitraf.tasks.trick_classifier.video_mae import (
    VideoMAEEvalConfig,
    run_evaluation as run_video_mae_evaluation,
)


EvaluationBuilder = Callable[[DictConfig, str], Any]
EvaluationRunner = Callable[[Any], None]


def _require_model_id(model_id: str | None) -> str:
    if not model_id:
        raise ValueError(
            "model_id must be provided (e.g. `python scripts/eval.py model_id=MyModel/1`)."
        )
    return model_id


def _build_pose_tcn_eval_config(cfg: DictConfig, model_id: str) -> PoseTCNEvalConfig:
    device = "cuda" if cfg.model.accelerator == "gpu" else cfg.model.accelerator

    return PoseTCNEvalConfig(
        model_uri=f"models:/{model_id}",
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.paths.vocab_path,
        target_col=cfg.task.target_column,
        poses_dir=cfg.model.poses_dir,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.sampling_dist,
        device=device,
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        top_k_worst=cfg.top_k_worst,
    )


def _build_video_mae_eval_config(cfg: DictConfig, model_id: str) -> VideoMAEEvalConfig:
    data_dir = Path(cfg.paths.data_dir)

    return VideoMAEEvalConfig(
        backbone=cfg.model.backbone,
        model_uri=f"models:/{model_id}",
        manifests_dir=cfg.task.manifests_dir,
        clips_dir=data_dir / "clips",
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.sampling_dist,
        device=cfg.model.device,
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        experiment_name=cfg.experiment_name,
        top_k_worst=cfg.top_k_worst,
    )


EVALUATION_TARGETS: dict[tuple[str, str], tuple[EvaluationBuilder, EvaluationRunner]] = {
    ("trick_classification", "pose_tcn"): (
        _build_pose_tcn_eval_config,
        run_pose_tcn_evaluation,
    ),
    ("trick_classification", "video_mae"): (
        _build_video_mae_eval_config,
        run_video_mae_evaluation,
    ),
}


@main(config_path="../configs", config_name="eval", version_base=None)
def run(cfg: DictConfig) -> None:
    load_dotenv()
    setup_logging()

    model_id = _require_model_id(cfg.model_id)

    key = (cfg.task.name, cfg.model.name)

    target = EVALUATION_TARGETS.get(key)
    
    if target is None:
        available = ", ".join(f"{task}/{model}" for task, model in EVALUATION_TARGETS)
        raise RuntimeError(
            f"No evaluation target registered for task='{cfg.task.name}', "
            f"model='{cfg.model.name}'. Available combinations: {available or 'none'}."
        )

    builder, runner = target
    eval_cfg = builder(cfg, model_id)

    logger.info(
        f"Starting evaluation for task='{cfg.task.name}' model='{cfg.model.name}' (run: {cfg.run_name})"
    )

    runner(eval_cfg)

    logger.info(
        f"Finished evaluation for task='{cfg.task.name}' model='{cfg.model.name}'."
    )


if __name__ == "__main__":
    run()
