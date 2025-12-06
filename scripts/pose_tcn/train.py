"""CLI entrypoint for inspecting pose TCN data."""

from hydra import main
from omegaconf import DictConfig
from dotenv import load_dotenv

from aitraf.logging import setup_logging
from aitraf.pose_tcn.training import PoseTCNTrainingConfig, run_training


@main(config_path="../../configs", config_name="pose_tcn", version_base=None)
def run(cfg: DictConfig) -> None:
    load_dotenv()
    setup_logging()

    training_cfg = PoseTCNTrainingConfig(
        manifests_dir=cfg.pose_tcn.manifests_dir,
        poses_dir=cfg.pose_tcn.poses_dir,
        batch_size=cfg.pose_tcn.batch_size,
        num_workers=cfg.pose_tcn.num_workers,
        sample_frames=cfg.pose_tcn.sample_frames,
        sampling_dist=cfg.pose_tcn.sampling_dist,
    )

    run_training(training_cfg)


if __name__ == "__main__":
    run()
