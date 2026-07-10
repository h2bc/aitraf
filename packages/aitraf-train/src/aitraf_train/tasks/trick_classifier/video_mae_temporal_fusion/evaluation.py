"""VideoMAE temporal-fusion evaluation for trick classification."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mlflow
from datasets import load_dataset
from dotenv import load_dotenv
from mlflow.data import from_huggingface
from transformers import Trainer, TrainingArguments

from aitraf_ml_core.loading import load_mlflow_torch_model
from aitraf_ml_core.pre_processing.paths import video_feature_cache_dir
from aitraf_train.logging import logger
from aitraf_train.metrics import (
    EvalModel,
    EvalSet,
    accuracy,
    calc_metrics_for_models,
    compute_dummy_classification_pred_ids,
    compute_pred_confidences,
    compute_pred_ids,
    f1_macro,
    flatten_metrics_report,
    get_confusion_matrix_figure,
    get_per_class_f1_figure,
    get_target_distribution_figure,
    get_top_k_worst_misses,
    metrics_to_df,
)
from aitraf_train.data.labels import (
    build_label_transform,
    load_target_label_mappings,
)
from aitraf_ml_core.processing.models.video_mae_temporal_fusion import (
    process_temporal_fusion_feature_sample,
)
from aitraf_train.data.collate import build_collate
from aitraf_train.tracking import (
    build_prediction_rows,
    build_training_params,
    log_test_predictions,
    log_train_predictions,
    params_to_df,
)
from aitraf_train.tracking.models.video_mae_temporal_fusion import TRAINING_PARAM_MAP


@dataclass
class VideoMaeTemporalFusionTrickClassificationEvalCfg:
    model_uri: str
    backbone: str
    manifests_dir: Path | str
    vocab_path: Path | str
    features_dir: Path | str
    batch_size: int
    num_workers: int
    sample_frames: int
    num_clips: int
    sampling_dist: str
    device: str
    output_dir: Path | str
    run_name: str
    experiment_name: str
    top_k_worst: int | None

    def __post_init__(self) -> None:
        self.manifests_dir = Path(self.manifests_dir)
        self.vocab_path = Path(self.vocab_path)
        self.features_dir = Path(self.features_dir)
        self.output_dir = Path(self.output_dir)


def run_evaluation(config: VideoMaeTemporalFusionTrickClassificationEvalCfg) -> None:
    load_dotenv()

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(config.manifests_dir / "train.jsonl"),
            "test": str(config.manifests_dir / "test.jsonl"),
        },
    )
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    label_names, label2id, id2label = load_target_label_mappings(
        config.vocab_path, "trick"
    )
    loaded_model = load_mlflow_torch_model(config.model_uri)
    model = loaded_model.model.to(config.device)
    logger.info(
        f"VideoMAE temporal-fusion evaluation running on device: {next(model.parameters()).device}"
    )
    label_transform = build_label_transform(label2id)
    feature_cache_dir = video_feature_cache_dir(
        features_dir=config.features_dir,
        backbone=config.backbone,
        num_clips=config.num_clips,
        sample_frames=config.sample_frames,
        sampling_dist=config.sampling_dist,
    )
    process_fn = partial(
        process_temporal_fusion_feature_sample,
        feature_cache_dir=feature_cache_dir,
        label_key="trick",
        label_transform=label_transform,
    )
    data_collator = build_collate(process_fn)
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_eval_batch_size=config.batch_size,
        dataloader_num_workers=config.num_workers,
        remove_unused_columns=False,
        report_to=["mlflow"],
        run_name=config.run_name,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    source_train_run_id = loaded_model.run_id
    source_train_params = build_training_params(
        source_train_run_id, TRAINING_PARAM_MAP
    ) | {
        "eval_sampling_dist": config.sampling_dist,
    }

    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params(source_train_params)
        mlflow.log_table(params_to_df(source_train_params), "params_table.json")
        mlflow.log_input(from_huggingface(train_dataset, name="train"), context="train")
        mlflow.log_input(from_huggingface(test_dataset, name="test"), context="test")

        train_logits, train_labels, _ = trainer.predict(train_dataset)
        test_logits, test_labels, _ = trainer.predict(test_dataset)
        train_pred_ids = compute_pred_ids(train_logits)
        test_pred_ids = compute_pred_ids(test_logits)
        test_examples_df = test_dataset.to_pandas()

        log_train_predictions(
            build_prediction_rows(
                train_dataset.to_pandas(),
                predictions=train_pred_ids,
                labels=train_labels,
                id2label=id2label,
                confidences=compute_pred_confidences(train_logits),
            )
        )

        log_test_predictions(
            build_prediction_rows(
                test_examples_df,
                predictions=test_pred_ids,
                labels=test_labels,
                id2label=id2label,
                confidences=compute_pred_confidences(test_logits),
            )
        )

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
                    name="video_mae_temporal_fusion",
                    sets=[
                        EvalSet(
                            name="train",
                            predictions=train_pred_ids,
                            labels=train_labels,
                        ),
                        EvalSet(
                            name="test",
                            predictions=test_pred_ids,
                            labels=test_labels,
                        ),
                    ],
                ),
            ],
            eval_metrics=(accuracy, f1_macro),
        )
        mlflow.log_metrics(flatten_metrics_report(metrics_report))
        mlflow.log_table(metrics_to_df(metrics_report), "metrics_table.json")

        dist_fig = get_target_distribution_figure(
            test_pred_ids,
            test_labels,
            label_names,
            id2label,
        )
        mlflow.log_figure(dist_fig, "predicted_vs_actual_target_counts.png")

        cm_fig = get_confusion_matrix_figure(test_pred_ids, test_labels, label_names)
        mlflow.log_figure(cm_fig, "confusion_matrix.png")

        f1_fig = get_per_class_f1_figure(test_pred_ids, test_labels, label_names)
        mlflow.log_figure(f1_fig, "per_class_f1.png")

        misses = get_top_k_worst_misses(
            test_logits,
            test_labels,
            test_examples_df,
            id2label,
            top_k=config.top_k_worst,
        )
        if not misses.empty:
            mlflow.log_table(misses, "misses_summary.json")


__all__ = ["VideoMaeTemporalFusionTrickClassificationEvalCfg", "run_evaluation"]
