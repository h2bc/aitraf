"""Unified train + eval entrypoint dispatching per task/model combo."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from hydra import main
from omegaconf import DictConfig

from aitraf.logging import logger, setup_logging
from aitraf.tasks.trick_classifier.pose_tcn import (
    PoseTCNEvalConfig,
    PoseTcnTrickClassificationCfg,
    run_evaluation as run_pose_tcn_evaluation,
    run_training as run_pose_tcn_training,
)
from aitraf.tasks.trick_classifier.video_mae import (
    VideoMAEEvalConfig,
    VideoMaeTrickClassificationCfg,
    run_evaluation as run_video_mae_evaluation,
    run_training as run_video_mae_training,
)


TrainingBuilder = Callable[[DictConfig], Any]
EvaluationBuilder = Callable[[DictConfig, str], Any]
TrainingRunner = Callable[[Any], str]
EvaluationRunner = Callable[[Any], None]


def _build_pose_tcn_training_config(cfg: DictConfig) -> PoseTcnTrickClassificationCfg:
    return PoseTcnTrickClassificationCfg(
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.paths.vocab_path,
        target_col=cfg.task.target_column,
        poses_dir=cfg.model.poses_dir,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.sampling_dist,
        learning_rate=cfg.model.learning_rate,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        kernel_size=cfg.model.kernel_size,
        dropout=cfg.model.dropout,
        max_epochs=cfg.model.max_epochs,
        accelerator=cfg.model.accelerator,
        early_stopping_patience=cfg.model.early_stopping_patience,
        experiment_name=cfg.experiment_name,
        run_name=cfg.train_run_name,
        output_dir=cfg.train_output_dir,
        max_train_samples=cfg.max_samples,
    )


def _build_pose_tcn_eval_config(cfg: DictConfig, model_uri: str) -> PoseTCNEvalConfig:
    device = "cuda" if cfg.model.accelerator == "gpu" else cfg.model.accelerator

    return PoseTCNEvalConfig(
        model_uri=model_uri,
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
        run_name=cfg.eval_run_name,
        top_k_worst=cfg.top_k_worst,
    )


def _build_video_mae_training_config(
    cfg: DictConfig,
) -> VideoMaeTrickClassificationCfg:
    data_dir = Path(cfg.paths.data_dir)

    return VideoMaeTrickClassificationCfg(
        backbone=cfg.model.backbone,
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.paths.vocab_path,
        target_col=cfg.task.target_column,
        clips_dir=data_dir / "clips",
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.sampling_dist,
        device=cfg.model.device,
        output_dir=cfg.train_output_dir,
        epochs=cfg.model.epochs,
        experiment_name=cfg.experiment_name,
        run_name=cfg.train_run_name,
        freeze_backbone=cfg.model.freeze_backbone,
        model_cache_dir=cfg.model.model_cache_dir,
        max_train_samples=cfg.max_samples,
        early_stopping_patience=cfg.model.early_stopping_patience,
    )


def _build_video_mae_eval_config(cfg: DictConfig, model_uri: str) -> VideoMAEEvalConfig:
    data_dir = Path(cfg.paths.data_dir)

    return VideoMAEEvalConfig(
        backbone=cfg.model.backbone,
        model_uri=model_uri,
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.paths.vocab_path,
        target_col=cfg.task.target_column,
        clips_dir=data_dir / "clips",
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.sampling_dist,
        device=cfg.model.device,
        output_dir=cfg.eval_output_dir,
        run_name=cfg.eval_run_name,
        experiment_name=cfg.experiment_name,
        top_k_worst=cfg.top_k_worst,
    )


TRAIN_EVAL_TARGETS: dict[
    tuple[str, str],
    tuple[TrainingBuilder, TrainingRunner, EvaluationBuilder, EvaluationRunner],
] = {
    (
        "trick_classification",
        "pose_tcn",
    ): (
        _build_pose_tcn_training_config,
        run_pose_tcn_training,
        _build_pose_tcn_eval_config,
        run_pose_tcn_evaluation,
    ),
    (
        "trick_classification",
        "video_mae",
    ): (
        _build_video_mae_training_config,
        run_video_mae_training,
        _build_video_mae_eval_config,
        run_video_mae_evaluation,
    ),
}


@main(config_path="../configs", config_name="train_eval", version_base=None)
def run(cfg: DictConfig) -> None:
    load_dotenv()
    setup_logging()

    key = (cfg.task.name, cfg.model.name)

    target = TRAIN_EVAL_TARGETS.get(key)
    if target is None:
        available = ", ".join(f"{task}/{model}" for task, model in TRAIN_EVAL_TARGETS)
        raise RuntimeError(
            f"No train/eval target registered for task='{cfg.task.name}', "
            f"model='{cfg.model.name}'. Available combinations: {available or 'none'}."
        )

    build_train, run_train, build_eval, run_eval = target

    training_cfg = build_train(cfg)

    logger.info(
        f"Starting training for task='{cfg.task.name}' model='{cfg.model.name}' "
        f"(run: {cfg.train_run_name})"
    )

    model_uri = run_train(training_cfg)

    logger.info(
        f"Finished training for task='{cfg.task.name}' model='{cfg.model.name}'. "
        f"Model URI: {model_uri}"
    )

    eval_cfg = build_eval(cfg, model_uri)

    logger.info(
        f"Starting evaluation for task='{cfg.task.name}' model='{cfg.model.name}' "
        f"(run: {cfg.eval_run_name})"
    )

    run_eval(eval_cfg)

    logger.info(
        f"Finished evaluation for task='{cfg.task.name}' model='{cfg.model.name}'."
    )


if __name__ == "__main__":
    run()
