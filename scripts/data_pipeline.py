"""Hydra-managed data pipeline entrypoint."""

from __future__ import annotations

from hydra import main
from omegaconf import DictConfig
from aitraf.data.pull_ls import LabelStudioExportConfig, pull_label_studio
from aitraf.data.build_manifests import ManifestBuildConfig, build_manifests


@main(config_path="../configs", config_name="config", version_base=None)
def run(cfg: DictConfig) -> None:
    if cfg.tasks.pull_ls:
        pull_label_studio(
            LabelStudioExportConfig(
                output_path=cfg.label_studio.output_path,
                force=cfg.label_studio.force,
            )
        )
    if cfg.tasks.build_manifests:
        build_manifests(
            ManifestBuildConfig(
                input_path=cfg.manifests.input_path,
                output_dir=cfg.manifests.output_dir,
                val_ratio=cfg.manifests.val_ratio,
                test_ratio=cfg.manifests.test_ratio,
                seed=cfg.manifests.seed,
                force=cfg.manifests.force,
            )
        )


if __name__ == "__main__":
    run()
