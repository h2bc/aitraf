"""Hydra-managed data pipeline entrypoint."""

from hydra import main
from omegaconf import DictConfig
from dotenv import load_dotenv
from aitraf.data_ops.download_labels import LabelDownloadConfig, download_labels
from aitraf.data_ops.create_manifests import (
    ManifestBuildConfig,
    TaskConfig,
    create_manifests,
)
from aitraf.data_ops.download_clips import ClipDownloadConfig, download_clips
from aitraf.logging import setup_logging, heading
from aitraf.data_ops.pose_and_bbox_extraction import (
    PoseAndBBoxExtractionConfig,
    pose_and_bbox_extraction,
)
from aitraf.data_ops.download_ranks import RankDownloadConfig, download_ranks


@main(config_path="../configs", config_name="data_ops", version_base=None)
def run(cfg: DictConfig) -> None:
    load_dotenv()
    setup_logging()
    data_cfg = cfg.get("data_ops", cfg)

    if data_cfg.download_labels.enabled:
        heading("Download Labels")
        download_labels(
            LabelDownloadConfig(
                prefix=data_cfg.download_labels.prefix,
                output_path=data_cfg.download_labels.output_path,
                force=data_cfg.download_labels.force,
            )
        )
    else:
        heading("Skip label download (disabled)")

    if data_cfg.download_clips.enabled:
        heading("Download Clips")
        download_clips(
            ClipDownloadConfig(
                labels_path=data_cfg.download_clips.labels_path,
                output_dir=data_cfg.download_clips.output_dir,
                force=data_cfg.download_clips.force,
            )
        )
    else:
        heading("Skip clip download (disabled)")

    if data_cfg.pose_and_bbox_extraction.enabled:
        heading("Pose + BBox Extraction")
        pose_and_bbox_extraction(
            PoseAndBBoxExtractionConfig(
                clips_dir=data_cfg.pose_and_bbox_extraction.clips_dir,
                poses_dir=data_cfg.pose_and_bbox_extraction.poses_dir,
                boxes_dir=data_cfg.pose_and_bbox_extraction.boxes_dir,
                weights_path=data_cfg.pose_and_bbox_extraction.weights_path,
                device=data_cfg.pose_and_bbox_extraction.device,
                imgsz=data_cfg.pose_and_bbox_extraction.imgsz,
                conf=data_cfg.pose_and_bbox_extraction.conf,
                batch_size=data_cfg.pose_and_bbox_extraction.batch_size,
                max_det=data_cfg.pose_and_bbox_extraction.max_det,
                force=data_cfg.pose_and_bbox_extraction.force,
                limit=data_cfg.pose_and_bbox_extraction.limit,
            )
        )
    else:
        heading("Skip pose/bbox extraction (disabled)")

    if data_cfg.download_ranks.enabled:
        heading("Download Ranks")
        download_ranks(
            RankDownloadConfig(
                prefix=data_cfg.download_ranks.prefix,
                output_path=data_cfg.download_ranks.output_path,
                force=data_cfg.download_ranks.force,
            )
        )
    else:
        heading("Skip rank download (disabled)")

    if data_cfg.create_manifests.enabled:
        heading("Build Manifests")
        create_manifests(
            ManifestBuildConfig(
                input_path=data_cfg.create_manifests.input_path,
                output_dir=data_cfg.create_manifests.output_dir,
                val_ratio=data_cfg.create_manifests.val_ratio,
                test_ratio=data_cfg.create_manifests.test_ratio,
                force=data_cfg.create_manifests.force,
                tasks=[
                    TaskConfig(
                        name=task_cfg.name,
                        target_column=task_cfg.target_column,
                        video_col=task_cfg.video_col,
                        required_cols=task_cfg.required_cols,
                        stratify_col=task_cfg.stratify_col,
                        stratify_strategy=task_cfg.stratify_strategy,
                        manifests_dir=task_cfg.manifests_dir,
                    )
                    for task_cfg in data_cfg.create_manifests.tasks.values()
                ],
            )
        )
    else:
        heading("Skip manifest build (disabled)")


if __name__ == "__main__":
    run()
