import ast
from pathlib import Path


ML_CORE = Path(__file__).parents[1] / "src"


def test_ml_core_does_not_import_train_or_api() -> None:
    prohibited: list[tuple[Path, str]] = []
    for path in ML_CORE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = node.module if isinstance(node, ast.ImportFrom) else None
            if module and module.split(".", 1)[0] in {"aitraf_train", "aitraf_api"}:
                prohibited.append((path, module))
    assert prohibited == []
