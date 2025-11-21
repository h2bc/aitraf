"""CLI entrypoint for VideoMAE training experiments."""

from hydra import main
from omegaconf import DictConfig

from pathlib import Path

from aitraf.video_mae.training import VideoMAETrainingConfig, run_training


@main(config_path="../../configs", config_name="video_mae", version_base=None)
def run(cfg: DictConfig) -> None:
    training_cfg = VideoMAETrainingConfig(
        backbone=cfg.video_mae.backbone,
        manifests_dir=cfg.video_mae.manifests_dir,
        clips_dir=Path(cfg.paths.data_dir) / "clips",
        batch_size=cfg.video_mae.training.batch_size,
        num_workers=cfg.video_mae.training.num_workers,
        sample_frames=cfg.video_mae.training.sample_frames,
        device=cfg.video_mae.training.device,
        output_dir=cfg.video_mae.training.output_dir,
        epochs=cfg.video_mae.training.epochs,
        experiment_name=cfg.video_mae.training.experiment_name,
        run_name=cfg.video_mae.training.run_name,
    )

    run_training(training_cfg)


if __name__ == "__main__":
    run()
