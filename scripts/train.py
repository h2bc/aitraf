"""Unified training entrypoint dispatching per task/model combo."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from dotenv import load_dotenv
from hydra import main
from omegaconf import DictConfig

from aitraf.logging import logger, setup_logging
from aitraf.tasks.trick_classifier.pose_tcn import (
    PoseTcnTrickClassificationTrainCfg,
    run_training as run_pose_tcn_trick_classification_train,
)
from aitraf.tasks.trick_classifier.video_mae import (
    VideoMaeTrickClassificationTrainCfg,
    run_training as run_video_mae_trick_classification_train,
)
from aitraf.tasks.score_prediction.pose_tcn import (
    PoseTcnScorePredictionTrainCfg,
    run_training as run_pose_tcn_score_prediction_train,
)
from aitraf.tasks.score_prediction.video_mae import (
    VideoMaeScorePredictionTrainCfg,
    run_training as run_video_mae_score_prediction_train,
)
from aitraf.tasks.score_prediction_binary.video_mae import (
    VideoMaeScorePredictionBinaryTrainCfg,
    run_training as run_video_mae_score_prediction_binary_train,
)
from aitraf.tasks.score_prediction_pairwise.video_mae import (
    VideoMaeScorePredictionPairwiseTrainCfg,
    run_training as run_video_mae_score_prediction_pairwise_train,
)


def _build_video_mae_training_config(
    cfg: DictConfig,
) -> VideoMaeTrickClassificationTrainCfg:
    data_dir = Path(cfg.paths.data_dir)

    return VideoMaeTrickClassificationTrainCfg(
        task_name=cfg.task.name,
        model_name=cfg.model.name,
        backbone=cfg.model.backbone,
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
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


def _build_pose_tcn_training_config(
    cfg: DictConfig,
) -> PoseTcnTrickClassificationTrainCfg:
    return PoseTcnTrickClassificationTrainCfg(
        task_name=cfg.task.name,
        model_name=cfg.model.name,
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
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


def _build_pose_tcn_score_prediction_training_config(
    cfg: DictConfig,
) -> PoseTcnScorePredictionTrainCfg:
    return PoseTcnScorePredictionTrainCfg(
        task_name=cfg.task.name,
        model_name=cfg.model.name,
        manifests_dir=cfg.task.manifests_dir,
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


def _build_video_mae_score_prediction_training_config(
    cfg: DictConfig,
) -> VideoMaeScorePredictionTrainCfg:
    data_dir = Path(cfg.paths.data_dir)

    return VideoMaeScorePredictionTrainCfg(
        task_name=cfg.task.name,
        model_name=cfg.model.name,
        backbone=cfg.model.backbone,
        manifests_dir=cfg.task.manifests_dir,
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


def _build_video_mae_score_prediction_binary_training_config(
    cfg: DictConfig,
) -> VideoMaeScorePredictionBinaryTrainCfg:
    data_dir = Path(cfg.paths.data_dir)

    return VideoMaeScorePredictionBinaryTrainCfg(
        task_name=cfg.task.name,
        model_name=cfg.model.name,
        backbone=cfg.model.backbone,
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
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


def _build_video_mae_score_prediction_pairwise_training_config(
    cfg: DictConfig,
) -> VideoMaeScorePredictionPairwiseTrainCfg:
    data_dir = Path(cfg.paths.data_dir)

    return VideoMaeScorePredictionPairwiseTrainCfg(
        task_name=cfg.task.name,
        model_name=cfg.model.name,
        backbone=cfg.model.backbone,
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
        clips_dir=data_dir / "clips",
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.sampling_dist,
        device=cfg.model.device,
        output_dir=cfg.output_dir,
        model_cache_dir=cfg.model.model_cache_dir,
        epochs=cfg.model.epochs,
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        freeze_backbone=cfg.model.freeze_backbone,
        early_stopping_patience=cfg.model.early_stopping_patience,
        max_train_samples=cfg.max_samples,
    )


TRAINING_TARGETS: dict[tuple[str, str], Callable[[DictConfig], str]] = {
    (
        "trick_classification",
        "video_mae",
    ): lambda cfg: run_video_mae_trick_classification_train(
        _build_video_mae_training_config(cfg)
    ),
    (
        "trick_classification",
        "pose_tcn",
    ): lambda cfg: run_pose_tcn_trick_classification_train(
        _build_pose_tcn_training_config(cfg)
    ),
    ("score_prediction", "pose_tcn"): lambda cfg: run_pose_tcn_score_prediction_train(
        _build_pose_tcn_score_prediction_training_config(cfg)
    ),
    ("score_prediction", "video_mae"): lambda cfg: run_video_mae_score_prediction_train(
        _build_video_mae_score_prediction_training_config(cfg)
    ),
    (
        "score_prediction_binary",
        "video_mae",
    ): lambda cfg: run_video_mae_score_prediction_binary_train(
        _build_video_mae_score_prediction_binary_training_config(cfg)
    ),
    (
        "score_prediction_pairwise",
        "video_mae",
    ): lambda cfg: run_video_mae_score_prediction_pairwise_train(
        _build_video_mae_score_prediction_pairwise_training_config(cfg)
    ),
}


@main(config_path="../configs", config_name="train", version_base=None)
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

    logger.info(
        f"Starting training for task='{cfg.task.name}' model='{cfg.model.name}' (run: {cfg.run_name})"
    )

    model_uri = target(cfg)

    logger.info(
        f"Finished training for task='{cfg.task.name}' model='{cfg.model.name}'. Model URI: {model_uri}"
    )


if __name__ == "__main__":
    run()
