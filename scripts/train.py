"""Unified training entrypoint dispatching per task/model combo."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from hydra import main
from omegaconf import DictConfig

from aitraf.logging import logger, setup_logging
from aitraf.tasks.trick_classifier.pose_tcn import (
    PoseTCNTrainingConfig,
    run_training as run_pose_tcn_training,
)
from aitraf.tasks.trick_classifier.video_mae import (
    VideoMAETrainingConfig,
    run_training as run_video_mae_training,
)


TrainingBuilder = Callable[[DictConfig], Any]
TrainingRunner = Callable[[Any], str]


def _build_video_mae_training_config(cfg: DictConfig) -> VideoMAETrainingConfig:
    data_dir = Path(cfg.paths.data_dir)

    return VideoMAETrainingConfig(
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
        output_dir=cfg.output_dir,
        epochs=cfg.model.epochs,
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        freeze_backbone=cfg.model.freeze_backbone,
        model_cache_dir=cfg.model.model_cache_dir,
        max_train_samples=cfg.max_samples,
        early_stopping_patience=cfg.model.early_stopping_patience,
    )


def _build_pose_tcn_training_config(cfg: DictConfig) -> PoseTCNTrainingConfig:
    return PoseTCNTrainingConfig(
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
        run_name=cfg.run_name,
        output_dir=cfg.output_dir,
        max_train_samples=cfg.max_samples,
    )


TRAINING_TARGETS: dict[tuple[str, str], tuple[TrainingBuilder, TrainingRunner]] = {
    ("trick_classification", "video_mae"): (
        _build_video_mae_training_config,
        run_video_mae_training,
    ),
    ("trick_classification", "pose_tcn"): (
        _build_pose_tcn_training_config,
        run_pose_tcn_training,
    ),
}


@main(config_path="../configs_v2", config_name="train", version_base=None)
def run(cfg: DictConfig) -> None:
    load_dotenv()
    setup_logging()

    key = (cfg.task.name, cfg.model.name)
    target = TRAINING_TARGETS.get(key)
    if target is None:
        available = ", ".join(f"{task}/{model}" for task, model in TRAINING_TARGETS)
        raise RuntimeError(
            f"No training target registered for task='{cfg.task.name}', "
            f"model='{cfg.model.name}'. Available combinations: {available or 'none'}."
        )

    builder, runner = target
    training_cfg = builder(cfg)

    logger.info(
        f"Starting training for task='{cfg.task.name}' model='{cfg.model.name}' (run: {cfg.run_name})"
    )

    model_uri = runner(training_cfg)
    
    logger.info(
        f"Finished training for task='{cfg.task.name}' model='{cfg.model.name}'. Model URI: {model_uri}"
    )


if __name__ == "__main__":
    run()
