"""Run Pose TCN training followed by evaluation."""

from hydra import main
from omegaconf import DictConfig
from dotenv import load_dotenv

from aitraf.logging import setup_logging
from aitraf.tasks.trick_classifier.pose_tcn import (
    PoseTCNEvalConfig,
    PoseTCNTrainingConfig,
    run_evaluation,
    run_training,
)


@main(config_path="../../../configs/trick_classifier", config_name="pose_tcn", version_base=None)
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
        learning_rate=cfg.pose_tcn.training.learning_rate,
        hidden_dim=cfg.pose_tcn.training.hidden_dim,
        num_layers=cfg.pose_tcn.training.num_layers,
        kernel_size=cfg.pose_tcn.training.kernel_size,
        dropout=cfg.pose_tcn.training.dropout,
        max_epochs=cfg.pose_tcn.training.max_epochs,
        early_stopping_patience=cfg.pose_tcn.training.early_stopping_patience,
        accelerator=cfg.pose_tcn.accelerator,
        experiment_name=cfg.pose_tcn.experiment_name,
        run_name=cfg.pose_tcn.training.run_name,
        output_dir=cfg.pose_tcn.training.output_dir,
        max_train_samples=cfg.pose_tcn.training.max_train_samples,
    )

    model_uri = run_training(training_cfg)

    device = "cuda" if cfg.pose_tcn.accelerator == "gpu" else cfg.pose_tcn.accelerator

    eval_cfg = PoseTCNEvalConfig(
        model_uri=model_uri,
        manifests_dir=cfg.pose_tcn.manifests_dir,
        poses_dir=cfg.pose_tcn.poses_dir,
        batch_size=cfg.pose_tcn.batch_size,
        num_workers=cfg.pose_tcn.num_workers,
        sample_frames=cfg.pose_tcn.sample_frames,
        sampling_dist=cfg.pose_tcn.sampling_dist,
        device=device,
        experiment_name=cfg.pose_tcn.experiment_name,
        run_name=cfg.pose_tcn.evaluation.run_name,
        top_k_worst=cfg.pose_tcn.evaluation.top_k_worst,
    )

    run_evaluation(eval_cfg)


if __name__ == "__main__":
    run()
