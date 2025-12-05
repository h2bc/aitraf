"""Run VideoMAE training then evaluate the resulting model."""

from pathlib import Path

from hydra import main
from omegaconf import DictConfig
from dotenv import load_dotenv
from aitraf.video_mae.training import VideoMAETrainingConfig, run_training
from aitraf.video_mae.evaluation import VideoMAEEvalConfig, run_evaluation


@main(config_path="../../configs", config_name="video_mae", version_base=None)
def run(cfg: DictConfig) -> None:
    load_dotenv()

    training_cfg = VideoMAETrainingConfig(
        backbone=cfg.video_mae.backbone,
        manifests_dir=cfg.video_mae.manifests_dir,
        clips_dir=Path(cfg.paths.data_dir) / "clips",
        batch_size=cfg.video_mae.batch_size,
        num_workers=cfg.video_mae.num_workers,
        sample_frames=cfg.video_mae.sample_frames,
        sampling_dist=cfg.video_mae.sampling_dist,
        device=cfg.video_mae.device,
        output_dir=cfg.video_mae.training.output_dir,
        epochs=cfg.video_mae.training.epochs,
        experiment_name=cfg.video_mae.experiment_name,
        run_name=cfg.video_mae.training.run_name,
        freeze_backbone=cfg.video_mae.training.freeze_backbone,
        model_cache_dir=cfg.video_mae.model_cache_dir,
        max_train_samples=cfg.video_mae.training.max_train_samples,
    )

    model_uri = run_training(training_cfg)

    eval_cfg = VideoMAEEvalConfig(
        backbone=cfg.video_mae.backbone,
        model_uri=model_uri,
        manifests_dir=cfg.video_mae.manifests_dir,
        clips_dir=Path(cfg.paths.data_dir) / "clips",
        batch_size=cfg.video_mae.batch_size,
        num_workers=cfg.video_mae.num_workers,
        sample_frames=cfg.video_mae.sample_frames,
        sampling_dist=cfg.video_mae.sampling_dist,
        device=cfg.video_mae.device,
        output_dir=cfg.video_mae.evaluation.output_dir,
        run_name=cfg.video_mae.evaluation.run_name,
        experiment_name=cfg.video_mae.experiment_name,
    )

    run_evaluation(eval_cfg)


if __name__ == "__main__":
    run()
