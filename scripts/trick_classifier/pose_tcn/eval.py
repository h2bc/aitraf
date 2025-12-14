"""CLI entrypoint for Pose TCN evaluation."""

from dotenv import load_dotenv
from hydra import main
from omegaconf import DictConfig

from aitraf.tasks.trick_classifier.pose_tcn import PoseTCNEvalConfig, run_evaluation


@main(config_path="../../../configs/trick_classifier", config_name="pose_tcn", version_base=None)
def run(cfg: DictConfig) -> None:
    load_dotenv()

    model_id = cfg.pose_tcn.evaluation.model_id

    if not model_id:
        raise ValueError(
            "pose_tcn.evaluation.model_id must be set to a valid MLflow model ID."
        )

    device = "cuda" if cfg.pose_tcn.accelerator == "gpu" else cfg.pose_tcn.accelerator

    eval_cfg = PoseTCNEvalConfig(
        model_uri=f"models:/{model_id}",
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
