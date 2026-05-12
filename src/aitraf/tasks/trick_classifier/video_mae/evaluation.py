"""VideoMAE evaluation pipeline."""

from dataclasses import dataclass
from functools import partial
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from mlflow.data import from_huggingface
import mlflow
from mlflow import transformers as mlflow_transformers
from transformers import (
    TrainingArguments,
    Trainer,
)

from aitraf.metrics import (
    EvalModel,
    EvalSet,
    calc_metrics_for_models,
    accuracy,
    f1_macro,
    compute_dummy_classification_pred_ids,
    compute_pred_ids,
    flatten_metrics_report,
    get_confusion_matrix_figure,
    get_miss_sampling_figures,
    get_per_class_f1_figure,
    get_target_distribution_figure,
    get_top_k_worst_misses,
    metrics_to_df,
)
from aitraf.tracking import build_training_params, params_to_df
from aitraf.processing import load_target_label_mappings
from aitraf.processing.models.video_mae import process_sample
from aitraf.processing.utils import build_collate
from aitraf.logging import logger
from aitraf.tracking.models.video_mae import TRAINING_PARAM_MAP


@dataclass
class VideoMaeTrickClassificationEvalCfg:
    """Configuration for VideoMAE evaluation."""

    backbone: str
    model_uri: str
    manifests_dir: Path | str
    vocab_path: Path | str
    clips_dir: Path | str
    batch_size: int
    num_workers: int
    sample_frames: int
    device: str
    output_dir: Path | str
    run_name: str
    experiment_name: str
    sampling_dist: str
    top_k_worst: int

    def __post_init__(self) -> None:
        self.manifests_dir = Path(self.manifests_dir)
        self.vocab_path = Path(self.vocab_path)
        self.clips_dir = Path(self.clips_dir)
        self.output_dir = Path(self.output_dir)


def run_evaluation(config: VideoMaeTrickClassificationEvalCfg):
    """Evaluate a fine-tuned VideoMAE model."""

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

    components = mlflow_transformers.load_model(
        config.model_uri, return_type="components"
    )
    model = components["model"].to(config.device)
    logger.info(
        f"VideoMAE evaluation running on device: {next(model.parameters()).device}"
    )
    processor = components["image_processor"]

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_eval_batch_size=config.batch_size,
        dataloader_num_workers=config.num_workers,
        remove_unused_columns=False,
        report_to=["mlflow"],
        run_name=config.run_name,
    )

    process_fn = partial(
        process_sample,
        processor=processor,
        local_clips_dir=config.clips_dir,
        num_frames=config.sample_frames,
        sampling_dist=config.sampling_dist,
        label_key="trick",
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
    source_train_params = build_training_params(
        source_train_run_id, TRAINING_PARAM_MAP
    ) | {"eval_sampling_dist": config.sampling_dist}

    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params(source_train_params)
        params_df = params_to_df(source_train_params)
        mlflow.log_table(params_df, "params_table.json")
        mlflow.log_input(from_huggingface(train_dataset, name="train"), context="train")
        mlflow.log_input(from_huggingface(test_dataset, name="test"), context="test")

        train_pred_logits, train_label_ids, _ = trainer.predict(train_dataset)
        train_video_mae_pred_ids = compute_pred_ids(train_pred_logits)
        train_dummy_pred_ids = compute_dummy_classification_pred_ids(train_label_ids)

        test_pred_logits, test_label_ids, _ = trainer.predict(test_dataset)
        test_video_mae_pred_ids = compute_pred_ids(test_pred_logits)
        test_dummy_pred_ids = compute_dummy_classification_pred_ids(test_label_ids)
        metrics_report = calc_metrics_for_models(
            eval_models=[
                EvalModel(
                    name="dummy",
                    sets=[
                        EvalSet(
                            name="train",
                            predictions=train_dummy_pred_ids,
                            labels=train_label_ids,
                        ),
                        EvalSet(
                            name="test",
                            predictions=test_dummy_pred_ids,
                            labels=test_label_ids,
                        ),
                    ],
                ),
                EvalModel(
                    name="video_mae",
                    sets=[
                        EvalSet(
                            name="train",
                            predictions=train_video_mae_pred_ids,
                            labels=train_label_ids,
                        ),
                        EvalSet(
                            name="test",
                            predictions=test_video_mae_pred_ids,
                            labels=test_label_ids,
                        ),
                    ],
                ),
            ],
            eval_metrics=(
                accuracy,
                f1_macro,
            ),
        )

        all_metrics = flatten_metrics_report(metrics_report)

        mlflow.log_metrics(all_metrics)
        metrics_df = metrics_to_df(metrics_report)
        mlflow.log_table(metrics_df, "metrics_table.json")

        dist_fig = get_target_distribution_figure(
            test_video_mae_pred_ids,
            test_label_ids,
            label_names,
            id2label,
        )

        mlflow.log_figure(dist_fig, "predicted_vs_actual_target_counts.png")

        cm_fig = get_confusion_matrix_figure(
            test_video_mae_pred_ids,
            test_label_ids,
            label_names,
        )
        mlflow.log_figure(cm_fig, "confusion_matrix.png")

        f1_fig = get_per_class_f1_figure(
            test_video_mae_pred_ids,
            test_label_ids,
            label_names,
        )
        mlflow.log_figure(f1_fig, "per_class_f1.png")

        test_examples_df = test_dataset.to_pandas()

        all_misses = get_top_k_worst_misses(
            test_pred_logits,
            test_label_ids,
            test_examples_df,
            id2label,
        )

        if not all_misses.empty:
            mlflow.log_table(all_misses, "misses_summary.json")
            for artifact_file, figure in get_miss_sampling_figures(
                all_misses,
                clips_dir=config.clips_dir,
                num_frames=config.sample_frames,
                sampling_dist=config.sampling_dist,
            ):
                mlflow.log_figure(figure, artifact_file)
