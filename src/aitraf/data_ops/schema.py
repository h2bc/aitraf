from typing import ClassVar, Dict, Callable, Any
import json
from enum import StrEnum


class ManifestType(StrEnum):
    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"


class DataSchema:
    columns: ClassVar[tuple[str, ...]] = ()
    types: ClassVar[Dict[str, str]] = {}
    processors: ClassVar[Dict[str, Callable[[Any], Any]]] = {}
    categorical: ClassVar[tuple[str, ...]] = ()


class LabelsSchema(DataSchema):
    columns: ClassVar[tuple[str, ...]] = (
        "annotation_id",
        "annotator",
        "created_at",
        "id",
        "key_foot",
        "lead_time",
        "person",
        "trick",
        "updated_at",
        "video",
        "execution_score",
        "execution_explanation",
    )

    types: ClassVar[Dict[str, str]] = {
        "annotation_id": "string",
        "annotator": "string",
        "id": "string",
        "execution_score": "Int64",
        "lead_time": "Float64",
        "created_at": "string",
        "updated_at": "string",
        "key_foot": "string",
        "person": "string",
        "trick": "string",
        "video": "string",
        "execution_explanation": "string",
    }

    processors: ClassVar[Dict[str, Callable[[Any], Any]]] = {
        "execution_score": lambda x: (
            json.loads(x)[0]["rating"] if isinstance(x, str) else x
        )
    }

    video_col: ClassVar[str] = "video"


class ManifestsSchema(DataSchema):
    columns: ClassVar[tuple[str, ...]] = (
        "video_id",
        "s3_path",
        "trick",
        "key_foot",
        "person",
        "execution_score",
        "execution_explanation",
    )

    types: ClassVar[dict[str, str]] = {
        "video_id": "string",
        "s3_path": "string",
        "trick": "string",
        "key_foot": "string",
        "person": "string",
        "execution_score": "Float64",
        "execution_explanation": "string",
    }

    processors: ClassVar[Dict[str, Callable[[Any], Any]]] = {
        "execution_score": lambda x: x / 4
    }

    categorical: ClassVar[tuple[str, ...]] = ("trick", "key_foot", "person")
    numerical: ClassVar[tuple[str, ...]] = ("execution_score",)


class PairwiseLabelsSchema(DataSchema):
    columns: ClassVar[tuple[str, ...]] = (
        "annotation_id",
        "task_id",
        "created_at",
        "updated_at",
        "lead_time",
        "annotator_id",
        "annotator_email",
        "created_username",
        "source_s3_key",
        "left",
        "right",
        "trick",
        "pref",
    )

    types: ClassVar[dict[str, str]] = {
        "annotation_id": "Int64",
        "task_id": "Int64",
        "created_at": "string",
        "updated_at": "string",
        "lead_time": "Float64",
        "annotator_id": "Int64",
        "annotator_email": "string",
        "created_username": "string",
        "source_s3_key": "string",
        "left": "string",
        "right": "string",
        "trick": "string",
    }


class PairwiseManifestsSchema(DataSchema):
    columns: ClassVar[tuple[str, ...]] = (
        "annotation_id",
        "task_id",
        "trick",
        "pair_label",
        "left_video_id",
        "right_video_id",
        "left_s3_path",
        "right_s3_path",
    )

    types: ClassVar[dict[str, str]] = {
        "annotation_id": "Int64",
        "task_id": "Int64",
        "trick": "string",
        "pair_label": "string",
        "left_video_id": "string",
        "right_video_id": "string",
        "left_s3_path": "string",
        "right_s3_path": "string",
    }

    categorical: ClassVar[tuple[str, ...]] = ("trick", "pair_label")
