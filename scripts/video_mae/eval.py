"""CLI entrypoint for VideoMAE evaluation."""

from hydra import main
from omegaconf import DictConfig
from pathlib import Path
from dotenv import load_dotenv

from aitraf.video_mae.evaluation import VideoMAEEvalConfig, run_evaluation


@main(config_path="../../configs", config_name="video_mae", version_base=None)
def run(cfg: DictConfig) -> None:
    load_dotenv()

    if not cfg.video_mae.evaluation.model_uri:
        raise ValueError(
            "evaluation.model_uri must be set to a valid MLflow model URI."
        )

    eval_cfg = VideoMAEEvalConfig(
        backbone=cfg.video_mae.backbone,
        model_uri=cfg.video_mae.evaluation.model_uri,
        manifests_dir=cfg.video_mae.manifests_dir,
        clips_dir=Path(cfg.paths.data_dir) / "clips",
        batch_size=cfg.video_mae.batch_size,
        num_workers=cfg.video_mae.num_workers,
        sample_frames=cfg.video_mae.sample_frames,
        device=cfg.video_mae.device,
        output_dir=cfg.video_mae.evaluation.output_dir,
        run_name=cfg.video_mae.evaluation.run_name,
        experiment_name=cfg.video_mae.experiment_name,
    )

    run_evaluation(eval_cfg)


if __name__ == "__main__":
    run()
