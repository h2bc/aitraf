from dataclasses import dataclass
from typing import Tuple, Dict, Type, FrozenSet
from datetime import datetime


@dataclass(frozen=True)
class LabelsSchema:
    columns: Tuple[str, ...] = (
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
    )

    types: Dict[str, Type] = {
        "annotation_id": int,
        "annotator": int,
        "created_at": datetime,
        "id": int,
        "key_foot": str,
        "lead_time": float,
        "person": str,
        "trick": str,
        "updated_at": datetime,
        "video": str,
    }


@dataclass(frozen=True)
class ManifestSchema:
    columns: Tuple[str, ...] = (
        "video_id",
        "s3_path",
        "trick",
        "key_foot",
        "person",
        "score",
    )

    types: Dict[str, Type] = {
        "video_id": str,
        "s3_path": str,
        "trick": str,
        "key_foot": str,
        "person": str,
        "score": float,
    }

    categorical: FrozenSet[str] = frozenset(
        {
            "trick",
            "key_foot",
            "person",
        }
    )

    numerical: FrozenSet[str] = frozenset(
        {
            "score",
        }
    )
