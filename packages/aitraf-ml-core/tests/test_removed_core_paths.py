from importlib.util import find_spec


def test_old_core_ml_and_clip_paths_are_removed() -> None:
    for module in (
        "aitraf_core.inference",
        "aitraf_core.loading",
        "aitraf_core.pre_processing",
        "aitraf_core.processing",
        "aitraf_core.storage.clips",
    ):
        assert find_spec(module) is None
