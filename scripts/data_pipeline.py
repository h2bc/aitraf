"""Hydra-managed data pipeline entrypoint."""

from hydra import main
from omegaconf import DictConfig
from dotenv import load_dotenv
from aitraf.data_ops.download_labels import LabelStudioExportConfig, download_labels
from aitraf.data_ops.create_manifests import ManifestBuildConfig, create_manifests
from aitraf.data_ops.download_clips import ClipDownloadConfig, download_clips
from aitraf.logging import setup_logging, heading
from aitraf.data_ops.pose_and_bbox_extraction import (
    PoseAndBBoxExtractionConfig,
    pose_and_bbox_extraction,
)


@main(config_path="../configs", config_name="data_ops", version_base=None)
def run(cfg: DictConfig) -> None:
    load_dotenv()
    setup_logging()

    if cfg.data_ops.tasks.download_labels:
        heading("Download Labels")
        download_labels(
            LabelStudioExportConfig(
                output_path=cfg.data_ops.download_labels.output_path,
                force=cfg.data_ops.download_labels.force,
            )
        )
    else:
        heading("Skip label download (disabled)")

    if cfg.data_ops.tasks.download_clips:
        heading("Download Clips")
        download_clips(
            ClipDownloadConfig(
                labels_path=cfg.data_ops.download_clips.labels_path,
                output_dir=cfg.data_ops.download_clips.output_dir,
                force=cfg.data_ops.download_clips.force,
            )
        )
    else:
        heading("Skip clip download (disabled)")

    if cfg.data_ops.tasks.pose_and_bbox_extraction:
        heading("Pose + BBox Extraction")
        pose_and_bbox_extraction(
            PoseAndBBoxExtractionConfig(
                clips_dir=cfg.data_ops.pose_and_bbox_extraction.clips_dir,
                poses_dir=cfg.data_ops.pose_and_bbox_extraction.poses_dir,
                boxes_dir=cfg.data_ops.pose_and_bbox_extraction.boxes_dir,
                weights_path=cfg.data_ops.pose_and_bbox_extraction.weights_path,
                device=cfg.data_ops.pose_and_bbox_extraction.device,
                imgsz=cfg.data_ops.pose_and_bbox_extraction.imgsz,
                conf=cfg.data_ops.pose_and_bbox_extraction.conf,
                batch_size=cfg.data_ops.pose_and_bbox_extraction.batch_size,
                max_det=cfg.data_ops.pose_and_bbox_extraction.max_det,
                force=cfg.data_ops.pose_and_bbox_extraction.force,
                limit=cfg.data_ops.pose_and_bbox_extraction.limit,
            )
        )
    else:
        heading("Skip pose/bbox extraction (disabled)")

    if cfg.data_ops.tasks.create_manifests:
        heading("Build Manifests")
        create_manifests(
            ManifestBuildConfig(
                input_path=cfg.data_ops.create_manifests.input_path,
                output_dir=cfg.data_ops.create_manifests.output_dir,
                val_ratio=cfg.data_ops.create_manifests.val_ratio,
                test_ratio=cfg.data_ops.create_manifests.test_ratio,
                seed=cfg.data_ops.create_manifests.seed,
                force=cfg.data_ops.create_manifests.force,
            )
        )
    else:
        heading("Skip manifest build (disabled)")


if __name__ == "__main__":
    run()
