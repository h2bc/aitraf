"""Unified train + eval entrypoint dispatching per task/model combo."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from hydra import main
from omegaconf import DictConfig

from aitraf.logging import logger, setup_logging
from aitraf.tasks.trick_classifier.pose_tcn import (
    PoseTcnTrickClassificationEvalCfg,
    PoseTcnTrickClassificationTrainCfg,
    run_evaluation as run_pose_tcn_trick_classification_eval,
    run_training as run_pose_tcn_trick_classification_train,
)
from aitraf.tasks.trick_classifier.video_mae import (
    VideoMaeTrickClassificationEvalCfg,
    VideoMaeTrickClassificationTrainCfg,
    run_evaluation as run_video_mae_trick_classification_eval,
    run_training as run_video_mae_trick_classification_train,
)
from aitraf.tasks.score_prediction.pose_tcn import (
    PoseTcnScorePredictionTrainCfg,
    PoseTcnScorePredictionEvalCfg,
    run_evaluation as run_pose_tcn_score_prediction_eval,
    run_training as run_pose_tcn_score_prediction_train,
)
from aitraf.tasks.score_prediction.video_mae import (
    VideoMaeScorePredictionEvalCfg,
    VideoMaeScorePredictionTrainCfg,
    run_evaluation as run_video_mae_score_prediction_eval,
    run_training as run_video_mae_score_prediction_train,
)
from aitraf.tasks.score_prediction_binary.video_mae import (
    VideoMaeScorePredictionBinaryEvalCfg,
    VideoMaeScorePredictionBinaryTrainCfg,
    run_evaluation as run_video_mae_score_prediction_binary_eval,
    run_training as run_video_mae_score_prediction_binary_train,
)
from aitraf.tasks.score_prediction_binary.video_mae_temporal_fusion import (
    VideoMaeTemporalFusionScorePredictionBinaryEvalCfg,
    VideoMaeTemporalFusionScorePredictionBinaryTrainCfg,
    run_evaluation as run_video_mae_temporal_fusion_score_prediction_binary_eval,
    run_training as run_video_mae_temporal_fusion_score_prediction_binary_train,
)
from aitraf.tasks.score_prediction_pairwise.video_mae import (
    VideoMaeScorePredictionPairwiseEvalCfg,
    VideoMaeScorePredictionPairwiseTrainCfg,
    run_evaluation as run_video_mae_score_prediction_pairwise_eval,
    run_training as run_video_mae_score_prediction_pairwise_train,
)
from aitraf.tasks.score_prediction_ordinal.video_mae import (
    VideoMaeScorePredictionOrdinalEvalCfg,
    VideoMaeScorePredictionOrdinalTrainCfg,
    run_evaluation as run_video_mae_score_prediction_ordinal_eval,
    run_training as run_video_mae_score_prediction_ordinal_train,
)
from aitraf.tasks.score_prediction_ordinal.video_mae_temporal_fusion import (
    VideoMaeTemporalFusionScorePredictionOrdinalEvalCfg,
    VideoMaeTemporalFusionScorePredictionOrdinalTrainCfg,
    run_evaluation as run_video_mae_temporal_fusion_score_prediction_ordinal_eval,
    run_training as run_video_mae_temporal_fusion_score_prediction_ordinal_train,
)
from aitraf.tasks.trick_classifier.video_mae_temporal_fusion import (
    VideoMaeTemporalFusionTrickClassificationEvalCfg,
    VideoMaeTemporalFusionTrickClassificationTrainCfg,
    run_evaluation as run_video_mae_temporal_fusion_trick_classification_eval,
    run_training as run_video_mae_temporal_fusion_trick_classification_train,
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
        sampling_dist=cfg.model.train_sampling_dist,
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


def _build_pose_tcn_eval_config(
    cfg: DictConfig, model_uri: str
) -> PoseTcnTrickClassificationEvalCfg:
    device = "cuda" if cfg.model.accelerator == "gpu" else cfg.model.accelerator

    return PoseTcnTrickClassificationEvalCfg(
        model_uri=model_uri,
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
        poses_dir=cfg.model.poses_dir,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.eval_sampling_dist,
        device=device,
        experiment_name=cfg.experiment_name,
        run_name=cfg.eval_run_name,
        top_k_worst=cfg.top_k_worst,
    )


def _build_video_mae_training_config(
    cfg: DictConfig,
) -> VideoMaeTrickClassificationTrainCfg:
    return VideoMaeTrickClassificationTrainCfg(
        task_name=cfg.task.name,
        model_name=cfg.model.name,
        backbone=cfg.model.backbone,
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
        clips_dir=Path(cfg.paths.clips_dir),
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.train_sampling_dist,
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


def _build_video_mae_eval_config(
    cfg: DictConfig, model_uri: str
) -> VideoMaeTrickClassificationEvalCfg:
    return VideoMaeTrickClassificationEvalCfg(
        backbone=cfg.model.backbone,
        model_uri=model_uri,
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
        clips_dir=Path(cfg.paths.clips_dir),
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.eval_sampling_dist,
        device=cfg.model.device,
        output_dir=cfg.eval_output_dir,
        run_name=cfg.eval_run_name,
        experiment_name=cfg.experiment_name,
        top_k_worst=cfg.top_k_worst,
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
        sampling_dist=cfg.model.train_sampling_dist,
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


def _build_pose_tcn_score_prediction_eval_config(
    cfg: DictConfig, model_uri: str
) -> PoseTcnScorePredictionEvalCfg:
    device = "cuda" if cfg.model.accelerator == "gpu" else cfg.model.accelerator

    return PoseTcnScorePredictionEvalCfg(
        model_uri=model_uri,
        manifests_dir=cfg.task.manifests_dir,
        poses_dir=cfg.model.poses_dir,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.eval_sampling_dist,
        device=device,
        experiment_name=cfg.experiment_name,
        run_name=cfg.eval_run_name,
        top_k_worst=cfg.top_k_worst,
    )


def _build_video_mae_score_prediction_training_config(
    cfg: DictConfig,
) -> VideoMaeScorePredictionTrainCfg:
    return VideoMaeScorePredictionTrainCfg(
        task_name=cfg.task.name,
        model_name=cfg.model.name,
        backbone=cfg.model.backbone,
        manifests_dir=cfg.task.manifests_dir,
        clips_dir=Path(cfg.paths.clips_dir),
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.train_sampling_dist,
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


def _build_video_mae_score_prediction_eval_config(
    cfg: DictConfig, model_uri: str
) -> VideoMaeScorePredictionEvalCfg:
    return VideoMaeScorePredictionEvalCfg(
        model_uri=model_uri,
        manifests_dir=cfg.task.manifests_dir,
        clips_dir=Path(cfg.paths.clips_dir),
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.eval_sampling_dist,
        device=cfg.model.device,
        output_dir=cfg.eval_output_dir,
        run_name=cfg.eval_run_name,
        experiment_name=cfg.experiment_name,
        top_k_worst=cfg.top_k_worst,
    )


def _build_video_mae_score_prediction_binary_training_config(
    cfg: DictConfig,
) -> VideoMaeScorePredictionBinaryTrainCfg:
    return VideoMaeScorePredictionBinaryTrainCfg(
        task_name=cfg.task.name,
        model_name=cfg.model.name,
        backbone=cfg.model.backbone,
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
        clips_dir=Path(cfg.paths.clips_dir),
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.train_sampling_dist,
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


def _build_video_mae_score_prediction_binary_eval_config(
    cfg: DictConfig, model_uri: str
) -> VideoMaeScorePredictionBinaryEvalCfg:
    return VideoMaeScorePredictionBinaryEvalCfg(
        model_uri=model_uri,
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
        clips_dir=Path(cfg.paths.clips_dir),
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.eval_sampling_dist,
        device=cfg.model.device,
        output_dir=cfg.eval_output_dir,
        run_name=cfg.eval_run_name,
        experiment_name=cfg.experiment_name,
        top_k_worst=cfg.top_k_worst,
    )


def _build_video_mae_score_prediction_pairwise_training_config(
    cfg: DictConfig,
) -> VideoMaeScorePredictionPairwiseTrainCfg:
    return VideoMaeScorePredictionPairwiseTrainCfg(
        task_name=cfg.task.name,
        model_name=cfg.model.name,
        backbone=cfg.model.backbone,
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
        clips_dir=Path(cfg.paths.clips_dir),
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.train_sampling_dist,
        device=cfg.model.device,
        output_dir=cfg.train_output_dir,
        model_cache_dir=cfg.model.model_cache_dir,
        epochs=cfg.model.epochs,
        experiment_name=cfg.experiment_name,
        run_name=cfg.train_run_name,
        freeze_backbone=cfg.model.freeze_backbone,
        early_stopping_patience=cfg.model.early_stopping_patience,
        max_train_samples=cfg.max_samples,
    )


def _build_video_mae_score_prediction_pairwise_eval_config(
    cfg: DictConfig, model_uri: str
) -> VideoMaeScorePredictionPairwiseEvalCfg:
    return VideoMaeScorePredictionPairwiseEvalCfg(
        model_uri=model_uri,
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
        clips_dir=Path(cfg.paths.clips_dir),
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.eval_sampling_dist,
        device=cfg.model.device,
        output_dir=cfg.eval_output_dir,
        run_name=cfg.eval_run_name,
        experiment_name=cfg.experiment_name,
        top_k_worst=cfg.top_k_worst,
    )


def _build_video_mae_score_prediction_ordinal_training_config(
    cfg: DictConfig,
) -> VideoMaeScorePredictionOrdinalTrainCfg:
    return VideoMaeScorePredictionOrdinalTrainCfg(
        task_name=cfg.task.name,
        model_name=cfg.model.name,
        backbone=cfg.model.backbone,
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
        clips_dir=Path(cfg.paths.clips_dir),
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.train_sampling_dist,
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


def _build_video_mae_score_prediction_ordinal_eval_config(
    cfg: DictConfig, model_uri: str
) -> VideoMaeScorePredictionOrdinalEvalCfg:
    return VideoMaeScorePredictionOrdinalEvalCfg(
        model_uri=model_uri,
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
        clips_dir=Path(cfg.paths.clips_dir),
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.eval_sampling_dist,
        device=cfg.model.device,
        output_dir=cfg.eval_output_dir,
        run_name=cfg.eval_run_name,
        experiment_name=cfg.experiment_name,
        top_k_worst=cfg.top_k_worst,
    )


def _build_video_mae_temporal_fusion_training_config(
    cfg: DictConfig,
    *,
    config_cls,
):
    return config_cls(
        task_name=cfg.task.name,
        model_name=cfg.model.name,
        backbone=cfg.model.backbone,
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
        clips_dir=Path(cfg.paths.clips_dir),
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        num_clips=cfg.model.num_clips,
        sampling_dist=cfg.model.train_sampling_dist,
        device=cfg.model.device,
        output_dir=cfg.train_output_dir,
        epochs=cfg.model.epochs,
        experiment_name=cfg.experiment_name,
        run_name=cfg.train_run_name,
        freeze_backbone=cfg.model.freeze_backbone,
        model_cache_dir=cfg.model.model_cache_dir,
        max_train_samples=cfg.max_samples,
        early_stopping_patience=cfg.model.early_stopping_patience,
        fusion_layers=cfg.model.fusion_layers,
        fusion_heads=cfg.model.fusion_heads,
        fusion_dropout=cfg.model.fusion_dropout,
    )


def _build_video_mae_temporal_fusion_eval_config(
    cfg: DictConfig,
    model_uri: str,
    *,
    config_cls,
):
    return config_cls(
        model_uri=model_uri,
        backbone=cfg.model.backbone,
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
        clips_dir=Path(cfg.paths.clips_dir),
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        num_clips=cfg.model.num_clips,
        sampling_dist=cfg.model.eval_sampling_dist,
        device=cfg.model.device,
        output_dir=cfg.eval_output_dir,
        run_name=cfg.eval_run_name,
        experiment_name=cfg.experiment_name,
        top_k_worst=cfg.top_k_worst,
        model_cache_dir=cfg.model.model_cache_dir,
    )


TRAIN_EVAL_TARGETS: dict[
    tuple[str, str],
    tuple[Callable[[DictConfig], str], Callable[[DictConfig, str], None]],
] = {
    ("trick_classification", "pose_tcn"): (
        lambda cfg: run_pose_tcn_trick_classification_train(
            _build_pose_tcn_training_config(cfg)
        ),
        lambda cfg, model_uri: run_pose_tcn_trick_classification_eval(
            _build_pose_tcn_eval_config(cfg, model_uri)
        ),
    ),
    ("trick_classification", "video_mae"): (
        lambda cfg: run_video_mae_trick_classification_train(
            _build_video_mae_training_config(cfg)
        ),
        lambda cfg, model_uri: run_video_mae_trick_classification_eval(
            _build_video_mae_eval_config(cfg, model_uri)
        ),
    ),
    ("score_prediction", "pose_tcn"): (
        lambda cfg: run_pose_tcn_score_prediction_train(
            _build_pose_tcn_score_prediction_training_config(cfg)
        ),
        lambda cfg, model_uri: run_pose_tcn_score_prediction_eval(
            _build_pose_tcn_score_prediction_eval_config(cfg, model_uri)
        ),
    ),
    ("score_prediction", "video_mae"): (
        lambda cfg: run_video_mae_score_prediction_train(
            _build_video_mae_score_prediction_training_config(cfg)
        ),
        lambda cfg, model_uri: run_video_mae_score_prediction_eval(
            _build_video_mae_score_prediction_eval_config(cfg, model_uri)
        ),
    ),
    ("score_prediction_binary", "video_mae"): (
        lambda cfg: run_video_mae_score_prediction_binary_train(
            _build_video_mae_score_prediction_binary_training_config(cfg)
        ),
        lambda cfg, model_uri: run_video_mae_score_prediction_binary_eval(
            _build_video_mae_score_prediction_binary_eval_config(cfg, model_uri)
        ),
    ),
    ("score_prediction_pairwise", "video_mae"): (
        lambda cfg: run_video_mae_score_prediction_pairwise_train(
            _build_video_mae_score_prediction_pairwise_training_config(cfg)
        ),
        lambda cfg, model_uri: run_video_mae_score_prediction_pairwise_eval(
            _build_video_mae_score_prediction_pairwise_eval_config(cfg, model_uri)
        ),
    ),
    ("score_prediction_ordinal", "video_mae"): (
        lambda cfg: run_video_mae_score_prediction_ordinal_train(
            _build_video_mae_score_prediction_ordinal_training_config(cfg)
        ),
        lambda cfg, model_uri: run_video_mae_score_prediction_ordinal_eval(
            _build_video_mae_score_prediction_ordinal_eval_config(cfg, model_uri)
        ),
    ),
    ("trick_classification", "video_mae_temporal_fusion"): (
        lambda cfg: run_video_mae_temporal_fusion_trick_classification_train(
            _build_video_mae_temporal_fusion_training_config(
                cfg,
                config_cls=VideoMaeTemporalFusionTrickClassificationTrainCfg,
            )
        ),
        lambda cfg, model_uri: run_video_mae_temporal_fusion_trick_classification_eval(
            _build_video_mae_temporal_fusion_eval_config(
                cfg,
                model_uri,
                config_cls=VideoMaeTemporalFusionTrickClassificationEvalCfg,
            )
        ),
    ),
    ("score_prediction_binary", "video_mae_temporal_fusion"): (
        lambda cfg: run_video_mae_temporal_fusion_score_prediction_binary_train(
            _build_video_mae_temporal_fusion_training_config(
                cfg,
                config_cls=VideoMaeTemporalFusionScorePredictionBinaryTrainCfg,
            )
        ),
        lambda cfg, model_uri: run_video_mae_temporal_fusion_score_prediction_binary_eval(
            _build_video_mae_temporal_fusion_eval_config(
                cfg,
                model_uri,
                config_cls=VideoMaeTemporalFusionScorePredictionBinaryEvalCfg,
            )
        ),
    ),
    ("score_prediction_ordinal", "video_mae_temporal_fusion"): (
        lambda cfg: run_video_mae_temporal_fusion_score_prediction_ordinal_train(
            _build_video_mae_temporal_fusion_training_config(
                cfg,
                config_cls=VideoMaeTemporalFusionScorePredictionOrdinalTrainCfg,
            )
        ),
        lambda cfg, model_uri: run_video_mae_temporal_fusion_score_prediction_ordinal_eval(
            _build_video_mae_temporal_fusion_eval_config(
                cfg,
                model_uri,
                config_cls=VideoMaeTemporalFusionScorePredictionOrdinalEvalCfg,
            )
        ),
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

    run_train, run_eval = target

    logger.info(
        f"Starting training for task='{cfg.task.name}' model='{cfg.model.name}' "
        f"(run: {cfg.train_run_name})"
    )
    model_uri = run_train(cfg)

    logger.info(
        f"Finished training for task='{cfg.task.name}' model='{cfg.model.name}'. "
        f"Model URI: {model_uri}"
    )

    logger.info(
        f"Starting evaluation for task='{cfg.task.name}' model='{cfg.model.name}' "
        f"(run: {cfg.eval_run_name})"
    )
    run_eval(cfg, model_uri)

    logger.info(
        f"Finished evaluation for task='{cfg.task.name}' model='{cfg.model.name}'."
    )


if __name__ == "__main__":
    run()
