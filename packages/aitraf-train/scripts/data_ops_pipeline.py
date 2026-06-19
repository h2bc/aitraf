"""Hydra-managed data pipeline entrypoint."""

from hydra import main
from omegaconf import DictConfig
from dotenv import load_dotenv
from aitraf_train.data_ops.download_labels import LabelDownloadConfig, download_labels
from aitraf_train.data_ops.download_pairwise_labels import (
    PairwiseLabelDownloadConfig,
    download_pairwise_labels,
)
from aitraf_train.data_ops.download_clips import ClipDownloadConfig, download_clips
from aitraf_train.logging import setup_logging, heading
from aitraf_train.data_ops.pose_and_bbox_extraction import (
    PoseAndBBoxExtractionConfig,
    pose_and_bbox_extraction,
)
from aitraf_train.data_ops.video_mae_feature_extraction import (
    VideoMaeFeatureExtractionConfig,
    video_mae_feature_extraction,
)


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

    if data_cfg.video_mae_feature_extraction.enabled:
        heading("VideoMAE Feature Extraction")
        video_mae_feature_extraction(
            VideoMaeFeatureExtractionConfig(
                clips_dir=data_cfg.video_mae_feature_extraction.clips_dir,
                features_dir=data_cfg.video_mae_feature_extraction.features_dir,
                backbone=data_cfg.video_mae_feature_extraction.backbone,
                model_cache_dir=data_cfg.video_mae_feature_extraction.model_cache_dir,
                device=data_cfg.video_mae_feature_extraction.device,
                batch_size=data_cfg.video_mae_feature_extraction.batch_size,
                num_workers=data_cfg.video_mae_feature_extraction.num_workers,
                num_clips=data_cfg.video_mae_feature_extraction.num_clips,
                sample_frames=data_cfg.video_mae_feature_extraction.sample_frames,
                sampling_dist=data_cfg.video_mae_feature_extraction.sampling_dist,
                force=data_cfg.video_mae_feature_extraction.force,
                limit=data_cfg.video_mae_feature_extraction.limit,
            )
        )
    else:
        heading("Skip VideoMAE feature extraction (disabled)")

    if data_cfg.download_pairwise_labels.enabled:
        heading("Download Pairwise Labels")
        download_pairwise_labels(
            PairwiseLabelDownloadConfig(
                prefix=data_cfg.download_pairwise_labels.prefix,
                output_dir=data_cfg.download_pairwise_labels.output_dir,
                output_path=data_cfg.download_pairwise_labels.output_path,
                force=data_cfg.download_pairwise_labels.force,
            )
        )
    else:
        heading("Skip pairwise label download (disabled)")


if __name__ == "__main__":
    run()
