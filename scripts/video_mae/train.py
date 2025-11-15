"""CLI entrypoint for VideoMAE training experiments."""

from hydra import main
from omegaconf import DictConfig

from pathlib import Path

from aitraf.video_mae.training import VideoMAETrainingConfig, run_training


@main(config_path="../../configs", config_name="video_mae", version_base=None)
def run(cfg: DictConfig) -> None:
    training_cfg = VideoMAETrainingConfig(
        backbone=cfg.video_mae.backbone,
        manifest_path=cfg.video_mae.training.manifest,
        clips_dir=Path(cfg.paths.data_dir) / "clips",
        batch_size=cfg.video_mae.training.batch_size,
        num_workers=cfg.video_mae.training.num_workers,
        num_frames=cfg.video_mae.training.num_frames,
        device=cfg.video_mae.training.device,
        output_dir=cfg.video_mae.training.output_dir,
    )

    run_training(training_cfg)


if __name__ == "__main__":
    run()
