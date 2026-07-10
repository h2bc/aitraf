import json
from pathlib import Path

import pytest

from aitraf_core.utils import read_json, read_jsonl_records


def test_read_json_and_jsonl_require_objects(tmp_path: Path) -> None:
    json_path = tmp_path / "record.json"
    json_path.write_text(json.dumps({"id": "one"}), encoding="utf-8")
    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text('{"id": "one"}\n\n{"id": "two"}\n', encoding="utf-8")

    assert read_json(json_path) == {"id": "one"}
    assert read_jsonl_records(jsonl_path) == [{"id": "one"}, {"id": "two"}]


def test_read_json_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "records.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected JSON object"):
        read_json(path)
