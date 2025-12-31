from typing import ClassVar, Dict, Callable, Any
import json


class DataSchema:
    columns: ClassVar[tuple[str, ...]] = ()
    types: ClassVar[Dict[str, str]] = {}
    processors: ClassVar[Dict[str, Callable[[Any], Any]]] = {}


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

    input_col: ClassVar[str] = "video"


class ManifestSchema(DataSchema):
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
