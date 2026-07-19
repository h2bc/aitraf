import ast
import tomllib
from pathlib import Path


ROOT = Path(__file__).parents[3]
CORE = ROOT / "packages" / "aitraf-core"

# aitraf-core is copied into the serving image, so it must stay free of the ML
# training stack. The rendering helpers it owns need numpy/pillow/imageio/av,
# which are small pure-wheel dependencies; torch and friends remain excluded.
ALLOWED_DEPENDENCIES = ["av", "boto3", "imageio", "numpy", "pillow"]
PROHIBITED = {
    "aitraf_ml_core",
    "aitraf_train",
    "huggingface_hub",
    "kornia",
    "mlflow",
    "torch",
    "torchcodec",
    "transformers",
    "ultralytics",
}


def test_core_declares_only_lightweight_runtime_dependencies() -> None:
    manifest = tomllib.loads((CORE / "pyproject.toml").read_text(encoding="utf-8"))
    assert manifest["project"]["dependencies"] == ALLOWED_DEPENDENCIES


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
