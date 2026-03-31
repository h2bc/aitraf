"""Hydra-managed data pipeline entrypoint."""

from hydra import main
from omegaconf import DictConfig
from dotenv import load_dotenv
from aitraf.data_ops import schema
from aitraf.data_ops.download_labels import LabelDownloadConfig, download_labels
from aitraf.data_ops.create_manifests import (
    ManifestBuildConfig,
    TaskConfig,
    create_manifests,
)
from aitraf.data_ops.create_pairwise_manifests import (
    PairwiseManifestBuildConfig,
    PairwiseTaskConfig,
    create_pairwise_manifests,
)
from aitraf.data_ops.download_pairwise_labels import (
    PairwiseLabelDownloadConfig,
    download_pairwise_labels,
)
from aitraf.data_ops.download_clips import ClipDownloadConfig, download_clips
from aitraf.logging import setup_logging, heading
from aitraf.data_ops.pose_and_bbox_extraction import (
    PoseAndBBoxExtractionConfig,
    pose_and_bbox_extraction,
)


def _build_task_stratify_kwargs(task_cfg: DictConfig) -> dict[str, str]:
    
    stratify_col = task_cfg.get("stratify_col")
    
    if stratify_col is None:
        return {}
    
    return {
        "stratify_col": stratify_col,
        "stratify_strategy": task_cfg.stratify_strategy,
    }


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

    if data_cfg.create_manifests.enabled:
        heading("Build Manifests")
        pointwise_tasks = []
        pairwise_tasks = []

        for task_cfg in data_cfg.create_manifests.tasks.values():
            manifest_type_raw = task_cfg.get(
                "manifest_type", schema.ManifestType.POINTWISE.value
            )
            try:
                manifest_type = schema.ManifestType(manifest_type_raw)
            except ValueError as exc:
                supported = ", ".join(member.value for member in schema.ManifestType)
                raise RuntimeError(
                    f"Unsupported manifest_type '{manifest_type_raw}' for task '{task_cfg.name}'. "
                    f"Expected one of: {supported}"
                ) from exc

            if manifest_type == schema.ManifestType.POINTWISE:
                pointwise_tasks.append(
                    TaskConfig(
                        name=task_cfg.name,
                        video_col=task_cfg.video_col,
                        required_cols=task_cfg.required_cols,
                        manifests_dir=task_cfg.manifests_dir,
                        vocab_path=task_cfg.get("vocab_path"),
                        **_build_task_stratify_kwargs(task_cfg),
                    )
                )
                continue

            if manifest_type == schema.ManifestType.PAIRWISE:
                pairwise_tasks.append(
                    PairwiseTaskConfig(
                        name=task_cfg.name,
                        pairwise_labels_path=task_cfg.pairwise_labels_path,
                        labels_path=task_cfg.labels_path,
                        pairwise_labels_required_cols=task_cfg.pairwise_labels_required_cols,
                        labels_required_cols=task_cfg.labels_required_cols,
                        manifests_dir=task_cfg.manifests_dir,
                        vocab_path=task_cfg.get("vocab_path"),
                        **_build_task_stratify_kwargs(task_cfg),
                    )
                )
                continue

        if pointwise_tasks:
            create_manifests(
                ManifestBuildConfig(
                    input_path=data_cfg.create_manifests.input_path,
                    output_dir=data_cfg.create_manifests.output_dir,
                    val_ratio=data_cfg.create_manifests.val_ratio,
                    test_ratio=data_cfg.create_manifests.test_ratio,
                    split_seed=data_cfg.create_manifests.split_seed,
                    force=data_cfg.create_manifests.force,
                    tasks=pointwise_tasks,
                )
            )

        if pairwise_tasks:
            create_pairwise_manifests(
                PairwiseManifestBuildConfig(
                    output_dir=data_cfg.create_manifests.output_dir,
                    val_ratio=data_cfg.create_manifests.val_ratio,
                    test_ratio=data_cfg.create_manifests.test_ratio,
                    split_seed=data_cfg.create_manifests.split_seed,
                    force=data_cfg.create_manifests.force,
                    tasks=pairwise_tasks,
                )
            )
    else:
        heading("Skip manifest build (disabled)")


if __name__ == "__main__":
    run()
