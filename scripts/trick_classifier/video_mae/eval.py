"""CLI entrypoint for VideoMAE evaluation."""

from hydra import main
from omegaconf import DictConfig
from pathlib import Path
from dotenv import load_dotenv

from aitraf.tasks.trick_classifier.video_mae import VideoMAEEvalConfig, run_evaluation


@main(config_path="../../../configs", config_name="video_mae", version_base=None)
def run(cfg: DictConfig) -> None:
    load_dotenv()

    model_id = cfg.video_mae.evaluation.model_id

    if not model_id:
        raise ValueError(
            "video_mae.evaluation.model_uri must be set to a valid MLflow model URI."
        )

    eval_cfg = VideoMAEEvalConfig(
        backbone=cfg.video_mae.backbone,
        model_uri=f"models:/{model_id}",
        manifests_dir=cfg.video_mae.manifests_dir,
        clips_dir=Path(cfg.paths.data_dir) / "clips",
        batch_size=cfg.video_mae.batch_size,
        num_workers=cfg.video_mae.num_workers,
        sample_frames=cfg.video_mae.sample_frames,
        sampling_dist=cfg.video_mae.sampling_dist,
        device=cfg.video_mae.device,
        output_dir=cfg.video_mae.training.output_dir,
        run_name=cfg.video_mae.evaluation.run_name,
        experiment_name=cfg.video_mae.experiment_name,
        top_k_worst=cfg.video_mae.evaluation.top_k_worst,
    )

    run_evaluation(eval_cfg)


if __name__ == "__main__":
    run()
