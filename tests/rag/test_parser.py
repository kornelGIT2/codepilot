from pathlib import Path

from app.services.genAI.rag.parser import parse_repo


def test_parser_uses_python_ast_chunking(tmp_path: Path):
    sample = tmp_path / "service.py"
    sample.write_text(
        """
import math

class UserService:
    def greet(self, name: str) -> str:
        return f\"Hello {name}\"


def compute_area(radius: float) -> float:
    return math.pi * radius * radius
""".strip(),
        encoding="utf-8",
    )

    chunks = parse_repo(str(tmp_path), extensions=[".py"])

    assert chunks
    assert any(c["metadata"]["chunk_type"] == "ast_node" for c in chunks)

    class_chunk = next(c for c in chunks if c["metadata"].get("symbol_name") == "UserService")
    function_chunk = next(c for c in chunks if c["metadata"].get("symbol_name") == "compute_area")

    assert class_chunk["metadata"]["node_type"] == "ClassDef"
    assert "UserService" in class_chunk["metadata"]["defined_symbols"]
    assert function_chunk["metadata"]["node_type"] == "FunctionDef"
    assert "compute_area" in function_chunk["metadata"]["defined_symbols"]


def test_parser_resolves_relative_imports_into_module_keys(tmp_path: Path):
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    (pkg_dir / "utils.py").write_text(
        """
def normalize_name(value: str) -> str:
    return value.strip().lower()
""".strip(),
        encoding="utf-8",
    )

    (pkg_dir / "service.py").write_text(
        """
from .utils import normalize_name


def process(name: str) -> str:
    return normalize_name(name)
""".strip(),
        encoding="utf-8",
    )

    chunks = parse_repo(str(tmp_path), extensions=[".py"])
    service_chunks = [c for c in chunks if c["metadata"]["file_path"].endswith("pkg/service.py")]

    assert service_chunks
    assert any("pkg/utils" in c["metadata"].get("resolved_imports", []) for c in service_chunks)
