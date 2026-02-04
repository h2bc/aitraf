"""Hydra-managed label ops pipeline entrypoint."""

from hydra import main
from omegaconf import DictConfig
from dotenv import load_dotenv

from aitraf.data_ops.download_labels import LabelStudioExportConfig, download_labels
from aitraf.label_ops.create_pairs import PairGenerationConfig, create_pairs
from aitraf.logging import setup_logging, heading


@main(config_path="../configs", config_name="label_ops", version_base=None)
def run(cfg: DictConfig) -> None:
    load_dotenv()
    setup_logging()
    ops_cfg = cfg.get("label_ops", cfg)

    if ops_cfg.download_labels.enabled:
        heading("Download Labels")
        download_labels(
            LabelStudioExportConfig(
                output_path=ops_cfg.download_labels.output_path,
                force=ops_cfg.download_labels.force,
            )
        )
    else:
        heading("Skip label download (disabled)")

    if ops_cfg.create_pairs.enabled:
        heading("Create Pair Files")
        create_pairs(
            PairGenerationConfig(
                labels_path=ops_cfg.create_pairs.labels_path,
                output_dir=ops_cfg.create_pairs.output_dir,
                k_per_video=ops_cfg.create_pairs.k_per_video,
                force=ops_cfg.create_pairs.force,
            )
        )
    else:
        heading("Skip pair creation (disabled)")


if __name__ == "__main__":
    run()
