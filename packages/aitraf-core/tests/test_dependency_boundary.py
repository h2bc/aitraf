import ast
import tomllib
from pathlib import Path


ROOT = Path(__file__).parents[3]
CORE = ROOT / "packages" / "aitraf-core"
PROHIBITED = {
    "aitraf_ml_core",
    "aitraf_train",
    "av",
    "huggingface_hub",
    "kornia",
    "mlflow",
    "numpy",
    "torch",
    "torchcodec",
    "transformers",
}


def test_core_declares_only_boto3_runtime_dependency() -> None:
    manifest = tomllib.loads((CORE / "pyproject.toml").read_text(encoding="utf-8"))
    assert manifest["project"]["dependencies"] == ["boto3"]


def test_core_source_does_not_import_ml_runtime() -> None:
    imports: set[str] = set()
    for path in (CORE / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
    assert imports.isdisjoint(PROHIBITED)
