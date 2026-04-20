"""Raw source schemas shared by ingestion/downloading steps."""

from typing import ClassVar, Dict, Callable, Any
import json


class LabelsSchema:
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


class PairwiseLabelsSchema:
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
