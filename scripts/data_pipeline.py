"""Hydra-managed data pipeline entrypoint."""

from __future__ import annotations

from hydra import main
from omegaconf import DictConfig
from aitraf.data.download_labels import LabelStudioExportConfig, download_labels
from aitraf.data.create_manifests import ManifestBuildConfig, create_manifests


@main(config_path="../configs", config_name="config", version_base=None)
def run(cfg: DictConfig) -> None:
    if cfg.data.tasks.download_labels:
        download_labels(
            LabelStudioExportConfig(
                output_path=cfg.data.download_labels.output_path,
                force=cfg.data.download_labels.force,
            )
        )
    if cfg.data.tasks.create_manifests:
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


if __name__ == "__main__":
    run()
