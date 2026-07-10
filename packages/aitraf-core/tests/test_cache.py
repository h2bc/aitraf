from pathlib import Path

from aitraf_core.cache import with_file_cache


def test_with_file_cache_computes_and_saves_missing_value(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    saved: list[tuple[Path, str]] = []

    value = with_file_cache(
        path=path,
        compute=lambda: "computed",
        save=lambda target, result: saved.append((target, result)),
    )

    assert value == "computed"
    assert saved == [(path, "computed")]


def test_with_file_cache_loads_existing_value(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    path.write_text("cached", encoding="utf-8")

    assert with_file_cache(
        path=path,
        compute=lambda: "computed",
        load=lambda target: target.read_text(encoding="utf-8"),
    ) == "cached"
