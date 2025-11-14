"""Hydra-managed data pipeline entrypoint."""

from hydra import main
from omegaconf import DictConfig
from aitraf.data.download_labels import LabelStudioExportConfig, download_labels
from aitraf.data.create_manifests import ManifestBuildConfig, create_manifests
from aitraf.data.download_clips import ClipDownloadConfig, download_clips
from aitraf.logging import setup_logging, heading


@main(config_path="../configs", config_name="data", version_base=None)
def run(cfg: DictConfig) -> None:
    setup_logging()

    if cfg.data.tasks.download_labels:
        heading("Download Labels")
        download_labels(
            LabelStudioExportConfig(
                output_path=cfg.data.download_labels.output_path,
                force=cfg.data.download_labels.force,
            )
        )
    else:
        heading("Skip label download (disabled)")

    if cfg.data.tasks.download_clips:
        heading("Download Clips")
        download_clips(
            ClipDownloadConfig(
                labels_path=cfg.data.download_clips.labels_path,
                output_dir=cfg.data.download_clips.output_dir,
                force=cfg.data.download_clips.force,
            )
        )
    else:
        heading("Skip clip download (disabled)")

    if cfg.data.tasks.create_manifests:
        heading("Build Manifests")
        create_manifests(
            ManifestBuildConfig(
                input_path=cfg.data.create_manifests.input_path,
                output_dir=cfg.data.create_manifests.output_dir,
                val_ratio=cfg.data.create_manifests.val_ratio,
                test_ratio=cfg.data.create_manifests.test_ratio,
                seed=cfg.data.create_manifests.seed,
                force=cfg.data.create_manifests.force,
            )
        )
    else:
        heading("Skip manifest build (disabled)")


if __name__ == "__main__":
    run()
