"""VideoMAE evaluation pipeline for pairwise ranking."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow import transformers as mlflow_transformers
from mlflow.data import from_pandas
from transformers import EvalPrediction, Trainer, TrainingArguments

from aitraf.datasets.score_prediction_rank import ScorePredictionRankDataset
from aitraf.logging import logger
from aitraf.metrics import build_pairwise_metrics
from aitraf.models import PairwiseRanker
from aitraf.processing import load_target_label_mappings
from aitraf.processing.models.video_mae import process_pair_sample
from aitraf.processing.utils import build_collate


@dataclass
class VideoMaeScorePredictionRankEvalCfg:
    """Configuration for evaluating VideoMAE pairwise ranking."""

    model_uri: str
    manifests_dir: Path | str
    vocab_path: Path | str
    clips_dir: Path | str
    batch_size: int
    num_workers: int
    sample_frames: int
    sampling_dist: str
    device: str
    output_dir: Path | str
    run_name: str
    experiment_name: str
    top_k_worst: int

    def __post_init__(self) -> None:
        self.manifests_dir = Path(self.manifests_dir)
        self.vocab_path = Path(self.vocab_path)
        self.clips_dir = Path(self.clips_dir)
        self.output_dir = Path(self.output_dir)


def run_evaluation(config: VideoMaeScorePredictionRankEvalCfg) -> None:
    """Evaluate a fine-tuned VideoMAE pairwise ranker."""

    load_dotenv()

    dataset = ScorePredictionRankDataset(
        manifests_dir=config.manifests_dir,
        split="test",
    )

    if len(dataset) == 0:
        raise RuntimeError("No test pairs found for score_prediction_rank.")

    _, label2id, _ = load_target_label_mappings(config.vocab_path, "pair_label")

    components = mlflow_transformers.load_model(
        config.model_uri, return_type="components"
    )
    scorer = components["model"].to(config.device)
    logger.info(
        f"VideoMAE evaluation running on device: {next(scorer.parameters()).device}"
    )
    processor = components["image_processor"]
    model = PairwiseRanker(scorer=scorer).to(config.device)

    compute_metrics = build_pairwise_metrics()

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_eval_batch_size=config.batch_size,
        dataloader_num_workers=config.num_workers,
        remove_unused_columns=False,
        report_to=["mlflow"],
        run_name=config.run_name,
    )

    process_fn = partial(
        process_pair_sample,
        processor=processor,
        local_clips_dir=config.clips_dir,
        num_frames=config.sample_frames,
        sampling_dist=config.sampling_dist,
        label_transform=lambda label: label2id[str(label)],
    )

    data_collator = build_collate(process_fn)

    def trainer_compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
        pair_logits, labels = prediction
        pred_labels = (np.asarray(pair_logits).squeeze(-1) >= 0).astype(int)
        return compute_metrics(pred_labels, np.asarray(labels))

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
        data_collator=data_collator,
        compute_metrics=trainer_compute_metrics,
    )

    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_input(
            from_pandas(pd.DataFrame(dataset.manifest_rows()), name="test"),
            context="test",
        )

        metrics = trainer.evaluate()
        _, label_ids, _ = trainer.predict(dataset)
        label_ids = np.asarray(label_ids).astype(int).reshape(-1)
        majority_label = int(np.bincount(label_ids).argmax())
        dummy_pred_labels = np.full_like(label_ids, fill_value=majority_label)
        dummy_metrics = {
            f"dummy_{key}": value
            for key, value in compute_metrics(dummy_pred_labels, label_ids).items()
        }
        tracked_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if key in {"eval_loss", "eval_accuracy"}
            and isinstance(value, (int, float))
        }
        mlflow.log_metrics(tracked_metrics)
        mlflow.log_metrics(dummy_metrics)


__all__ = ["VideoMaeScorePredictionRankEvalCfg", "run_evaluation"]
