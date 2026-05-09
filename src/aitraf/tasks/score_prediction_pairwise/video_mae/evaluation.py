"""VideoMAE evaluation pipeline for pairwise comparison."""

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
from transformers import Trainer, TrainingArguments

from aitraf.logging import logger
from aitraf.metrics import (
    EvalModel,
    EvalSet,
    accuracy,
    calc_metrics_for_models,
    compute_dummy_classification_pred_ids,
    flatten_metrics_report,
    metrics_to_df,
)
from aitraf.tracking import build_training_params, params_to_df
from aitraf.processing import load_target_label_mappings
from aitraf.processing.models.video_mae import process_pair_sample
from aitraf.processing.utils import build_collate
from aitraf.tracking.models.video_mae import TRAINING_PARAM_MAP
from ..dataset import ScorePredictionPairwiseDataset
from ..metrics import compute_pairwise_pred_labels
from .model import ScorePredictionPairwiseModel


@dataclass
class VideoMaeScorePredictionPairwiseEvalCfg:
    """Configuration for evaluating VideoMAE pairwise comparison."""

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


def run_evaluation(config: VideoMaeScorePredictionPairwiseEvalCfg) -> None:
    """Evaluate a fine-tuned VideoMAE pairwise comparator."""

    load_dotenv()

    train_dataset = ScorePredictionPairwiseDataset(
        manifests_dir=config.manifests_dir,
        split="train",
    )
    test_dataset = ScorePredictionPairwiseDataset(
        manifests_dir=config.manifests_dir,
        split="test",
    )

    if len(train_dataset) == 0:
        raise RuntimeError("No train pairs found for score_prediction_pairwise.")
    if len(test_dataset) == 0:
        raise RuntimeError("No test pairs found for score_prediction_pairwise.")

    _, label2id, _ = load_target_label_mappings(config.vocab_path, "pair_label")

    components = mlflow_transformers.load_model(
        config.model_uri, return_type="components"
    )
    scorer = components["model"].to(config.device)
    logger.info(
        f"VideoMAE evaluation running on device: {next(scorer.parameters()).device}"
    )
    processor = components["image_processor"]
    model = ScorePredictionPairwiseModel(scorer=scorer).to(config.device)

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

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    source_train_run_id = mlflow.models.get_model_info(config.model_uri).run_id
    source_train_params = build_training_params(source_train_run_id, TRAINING_PARAM_MAP) | {
        "sampling_dist": config.sampling_dist
    }

    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params(source_train_params)
        params_df = params_to_df(source_train_params)
        mlflow.log_table(params_df, "params_table.json")
        mlflow.log_input(
            from_pandas(pd.DataFrame(train_dataset.manifest_rows()), name="train"),
            context="train",
        )
        mlflow.log_input(
            from_pandas(pd.DataFrame(test_dataset.manifest_rows()), name="test"),
            context="test",
        )

        train_pair_logits, train_label_ids, _ = trainer.predict(train_dataset)
        train_labels = np.asarray(train_label_ids).reshape(-1).astype(int)
        train_video_mae_pred_labels = compute_pairwise_pred_labels(train_pair_logits)

        test_pair_logits, test_label_ids, _ = trainer.predict(test_dataset)
        test_labels = np.asarray(test_label_ids).reshape(-1).astype(int)
        test_video_mae_pred_labels = compute_pairwise_pred_labels(test_pair_logits)

        metrics_report = calc_metrics_for_models(
            eval_models=[
                EvalModel(
                    name="dummy",
                    sets=[
                        EvalSet(
                            name="train",
                            predictions=compute_dummy_classification_pred_ids(
                                train_labels
                            ),
                            labels=train_labels,
                        ),
                        EvalSet(
                            name="test",
                            predictions=compute_dummy_classification_pred_ids(
                                test_labels
                            ),
                            labels=test_labels,
                        ),
                    ],
                ),
                EvalModel(
                    name="video_mae",
                    sets=[
                        EvalSet(
                            name="train",
                            predictions=train_video_mae_pred_labels,
                            labels=train_labels,
                        ),
                        EvalSet(
                            name="test",
                            predictions=test_video_mae_pred_labels,
                            labels=test_labels,
                        ),
                    ],
                ),
            ],
            eval_metrics=(accuracy,),
        )
        all_metrics = flatten_metrics_report(metrics_report)

        mlflow.log_metrics(all_metrics)
        metrics_df = metrics_to_df(metrics_report)
        mlflow.log_table(metrics_df, "metrics_table.json")


__all__ = ["VideoMaeScorePredictionPairwiseEvalCfg", "run_evaluation"]
