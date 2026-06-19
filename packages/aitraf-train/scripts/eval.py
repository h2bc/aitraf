"""Unified evaluation entrypoint dispatching per task/model combo."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from hydra import main
from omegaconf import DictConfig

from aitraf_train.logging import logger, setup_logging
from aitraf_train.tasks.score_prediction_binary.video_mae_temporal_fusion import (
    VideoMaeTemporalFusionScorePredictionBinaryEvalCfg,
    run_evaluation as run_video_mae_temporal_fusion_score_prediction_binary_eval,
)
from aitraf_train.tasks.score_prediction_ordinal.video_mae_temporal_fusion import (
    VideoMaeTemporalFusionScorePredictionOrdinalEvalCfg,
    run_evaluation as run_video_mae_temporal_fusion_score_prediction_ordinal_eval,
)
from aitraf_train.tasks.score_prediction_ordinal.pose_tcn import (
    PoseTcnScorePredictionOrdinalEvalCfg,
    run_evaluation as run_pose_tcn_score_prediction_ordinal_eval,
)
from aitraf_train.tasks.trick_classifier.pose_tcn import (
    PoseTcnTrickClassificationEvalCfg,
    run_evaluation as run_pose_tcn_trick_classification_eval,
)
from aitraf_train.tasks.score_prediction.pose_tcn import (
    PoseTcnScorePredictionEvalCfg,
    run_evaluation as run_pose_tcn_score_prediction_eval,
)
from aitraf_train.tasks.score_prediction.video_mae import (
    VideoMaeScorePredictionEvalCfg,
    run_evaluation as run_video_mae_score_prediction_eval,
)
from aitraf_train.tasks.score_prediction_binary.video_mae import (
    VideoMaeScorePredictionBinaryEvalCfg,
    run_evaluation as run_video_mae_score_prediction_binary_eval,
)
from aitraf_train.tasks.score_prediction_pairwise.video_mae import (
    VideoMaeScorePredictionPairwiseEvalCfg,
    run_evaluation as run_video_mae_score_prediction_pairwise_eval,
)
from aitraf_train.tasks.score_prediction_ordinal.video_mae import (
    VideoMaeScorePredictionOrdinalEvalCfg,
    run_evaluation as run_video_mae_score_prediction_ordinal_eval,
)
from aitraf_train.tasks.trick_classifier.video_mae import (
    VideoMaeTrickClassificationEvalCfg,
    run_evaluation as run_video_mae_trick_classification_eval,
)
from aitraf_train.tasks.trick_classifier.video_mae_temporal_fusion import (
    VideoMaeTemporalFusionTrickClassificationEvalCfg,
    run_evaluation as run_video_mae_temporal_fusion_trick_classification_eval,
)


def _require_model_id(model_id: str | None) -> str:
    if not model_id:
        raise ValueError(
            "model_id must be provided (e.g. `task train:eval -- model_id=MyModel/1`)."
        )
    return model_id


def _build_model_uri(cfg: DictConfig) -> str:
    return f"models:/{cfg.model_id}"


def _build_pose_tcn_eval_config(cfg: DictConfig) -> PoseTcnTrickClassificationEvalCfg:
    return PoseTcnTrickClassificationEvalCfg(
        model_uri=_build_model_uri(cfg),
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
        poses_dir=cfg.model.poses_dir,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.eval_sampling_dist,
        device=cfg.model.device,
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        top_k_worst=cfg.top_k_worst,
    )


def _build_video_mae_eval_config(cfg: DictConfig) -> VideoMaeTrickClassificationEvalCfg:
    return VideoMaeTrickClassificationEvalCfg(
        backbone=cfg.model.backbone,
        model_uri=_build_model_uri(cfg),
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
        clips_dir=Path(cfg.paths.clips_dir),
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.eval_sampling_dist,
        device=cfg.model.device,
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        experiment_name=cfg.experiment_name,
        top_k_worst=cfg.top_k_worst,
    )


def _build_pose_tcn_score_prediction_eval_config(
    cfg: DictConfig,
) -> PoseTcnScorePredictionEvalCfg:
    return PoseTcnScorePredictionEvalCfg(
        model_uri=_build_model_uri(cfg),
        manifests_dir=cfg.task.manifests_dir,
        poses_dir=cfg.model.poses_dir,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.eval_sampling_dist,
        device=cfg.model.device,
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        top_k_worst=cfg.top_k_worst,
    )


def _build_pose_tcn_score_prediction_ordinal_eval_config(
    cfg: DictConfig,
) -> PoseTcnScorePredictionOrdinalEvalCfg:
    return PoseTcnScorePredictionOrdinalEvalCfg(
        model_uri=_build_model_uri(cfg),
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
        poses_dir=cfg.model.poses_dir,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.eval_sampling_dist,
        device=cfg.model.device,
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        top_k_worst=cfg.top_k_worst,
    )


def _build_video_mae_score_prediction_eval_config(
    cfg: DictConfig,
) -> VideoMaeScorePredictionEvalCfg:
    return VideoMaeScorePredictionEvalCfg(
        model_uri=_build_model_uri(cfg),
        manifests_dir=cfg.task.manifests_dir,
        clips_dir=Path(cfg.paths.clips_dir),
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.eval_sampling_dist,
        device=cfg.model.device,
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        experiment_name=cfg.experiment_name,
        top_k_worst=cfg.top_k_worst,
    )


def _build_video_mae_score_prediction_binary_eval_config(
    cfg: DictConfig,
) -> VideoMaeScorePredictionBinaryEvalCfg:
    return VideoMaeScorePredictionBinaryEvalCfg(
        model_uri=_build_model_uri(cfg),
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
        clips_dir=Path(cfg.paths.clips_dir),
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.eval_sampling_dist,
        device=cfg.model.device,
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        experiment_name=cfg.experiment_name,
        top_k_worst=cfg.top_k_worst,
    )


def _build_video_mae_score_prediction_pairwise_eval_config(
    cfg: DictConfig,
) -> VideoMaeScorePredictionPairwiseEvalCfg:
    return VideoMaeScorePredictionPairwiseEvalCfg(
        model_uri=_build_model_uri(cfg),
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
        clips_dir=Path(cfg.paths.clips_dir),
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.eval_sampling_dist,
        device=cfg.model.device,
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        experiment_name=cfg.experiment_name,
        top_k_worst=cfg.top_k_worst,
    )


def _build_video_mae_score_prediction_ordinal_eval_config(
    cfg: DictConfig,
) -> VideoMaeScorePredictionOrdinalEvalCfg:
    return VideoMaeScorePredictionOrdinalEvalCfg(
        model_uri=_build_model_uri(cfg),
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
        clips_dir=Path(cfg.paths.clips_dir),
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        sampling_dist=cfg.model.eval_sampling_dist,
        device=cfg.model.device,
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        experiment_name=cfg.experiment_name,
        top_k_worst=cfg.top_k_worst,
    )


def _build_video_mae_temporal_fusion_eval_config(
    cfg: DictConfig,
    *,
    config_cls,
):
    return config_cls(
        model_uri=_build_model_uri(cfg),
        backbone=cfg.model.backbone,
        manifests_dir=cfg.task.manifests_dir,
        vocab_path=cfg.task.vocab_path,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers,
        sample_frames=cfg.model.sample_frames,
        num_clips=cfg.model.num_clips,
        features_dir=Path(cfg.model.features_dir),
        sampling_dist=cfg.model.eval_sampling_dist,
        device=cfg.model.device,
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        experiment_name=cfg.experiment_name,
        top_k_worst=cfg.top_k_worst,
    )


EVALUATION_TARGETS: dict[tuple[str, str], Callable[[DictConfig], None]] = {
    (
        "trick_classification",
        "pose_tcn",
    ): lambda cfg: run_pose_tcn_trick_classification_eval(
        _build_pose_tcn_eval_config(cfg)
    ),
    (
        "trick_classification",
        "video_mae",
    ): lambda cfg: run_video_mae_trick_classification_eval(
        _build_video_mae_eval_config(cfg)
    ),
    ("score_prediction", "pose_tcn"): lambda cfg: run_pose_tcn_score_prediction_eval(
        _build_pose_tcn_score_prediction_eval_config(cfg)
    ),
    (
        "score_prediction_ordinal",
        "pose_tcn",
    ): lambda cfg: run_pose_tcn_score_prediction_ordinal_eval(
        _build_pose_tcn_score_prediction_ordinal_eval_config(cfg)
    ),
    ("score_prediction", "video_mae"): lambda cfg: run_video_mae_score_prediction_eval(
        _build_video_mae_score_prediction_eval_config(cfg)
    ),
    (
        "score_prediction_binary",
        "video_mae",
    ): lambda cfg: run_video_mae_score_prediction_binary_eval(
        _build_video_mae_score_prediction_binary_eval_config(cfg)
    ),
    (
        "score_prediction_pairwise",
        "video_mae",
    ): lambda cfg: run_video_mae_score_prediction_pairwise_eval(
        _build_video_mae_score_prediction_pairwise_eval_config(cfg)
    ),
    (
        "score_prediction_ordinal",
        "video_mae",
    ): lambda cfg: run_video_mae_score_prediction_ordinal_eval(
        _build_video_mae_score_prediction_ordinal_eval_config(cfg)
    ),
    (
        "trick_classification",
        "video_mae_temporal_fusion",
    ): lambda cfg: run_video_mae_temporal_fusion_trick_classification_eval(
        _build_video_mae_temporal_fusion_eval_config(
            cfg,
            config_cls=VideoMaeTemporalFusionTrickClassificationEvalCfg,
        )
    ),
    (
        "score_prediction_binary",
        "video_mae_temporal_fusion",
    ): lambda cfg: run_video_mae_temporal_fusion_score_prediction_binary_eval(
        _build_video_mae_temporal_fusion_eval_config(
            cfg,
            config_cls=VideoMaeTemporalFusionScorePredictionBinaryEvalCfg,
        )
    ),
    (
        "score_prediction_ordinal",
        "video_mae_temporal_fusion",
    ): lambda cfg: run_video_mae_temporal_fusion_score_prediction_ordinal_eval(
        _build_video_mae_temporal_fusion_eval_config(
            cfg,
            config_cls=VideoMaeTemporalFusionScorePredictionOrdinalEvalCfg,
        )
    ),
}


@main(config_path="../configs", config_name="eval", version_base=None)
def run(cfg: DictConfig) -> None:
    load_dotenv()
    setup_logging()

    cfg.model_id = _require_model_id(cfg.model_id)

    key = (cfg.task.name, cfg.model.name)

    target = EVALUATION_TARGETS.get(key)

    if target is None:
        available = ", ".join(f"{task}/{model}" for task, model in EVALUATION_TARGETS)
        raise RuntimeError(
            f"No evaluation target registered for task='{cfg.task.name}', "
            f"model='{cfg.model.name}'. Available combinations: {available or 'none'}."
        )

    logger.info(
        f"Starting evaluation for task='{cfg.task.name}' model='{cfg.model.name}' (run: {cfg.run_name})"
    )

    target(cfg)

    logger.info(
        f"Finished evaluation for task='{cfg.task.name}' model='{cfg.model.name}'."
    )


if __name__ == "__main__":
    run()
