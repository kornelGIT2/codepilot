import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

import networkx as nx

PYTHON_IMPORT_RE = re.compile(r"from\s+(?P<path>\.+[A-Za-z0-9_\.]+|\.+)\s+import")
TS_IMPORT_RE = re.compile(
    r"(?:from\s+['\"](?P<path1>\.[^'\"]+)['\"]|import\s+(?:['\"](?P<path2>\.[^'\"]+)['\"]|.*?\s+from\s+['\"](?P<path3>\.[^'\"]+)['\"]))"
)
VALID_SOURCE_EXTENSIONS = {".py", ".ts", ".tsx"}
VALID_TARGET_INDEX_FILES = ["__init__.py", "index.ts", "index.tsx"]


def parse_imports(file_path: str, content: str) -> List[str]:
 
    source_file = Path(file_path).resolve()
    imports: List[str] = []

    for import_path in _extract_relative_import_paths(content):
        resolved = _resolve_relative_import(source_file, import_path)
        if resolved is not None:
            imports.append(str(resolved))

    return imports


def build_dependency_graph(repo_path: str) -> nx.DiGraph:
   
    repo_root = Path(repo_path).resolve()
    graph = nx.DiGraph()

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in {"node_modules", "__pycache__", ".git"}]

        for filename in files:
            path = Path(root) / filename
            if path.suffix not in VALID_SOURCE_EXTENSIONS:
                continue

            source_path = path.resolve()
            graph.add_node(str(source_path))

            try:
                content = source_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            for import_target in parse_imports(str(source_path), content):
                target_path = Path(import_target)
                if not target_path.exists():
                    continue
                try:
                    target_path.relative_to(repo_root)
                except ValueError:
                    continue

                graph.add_node(str(target_path))
                graph.add_edge(str(source_path), str(target_path))

    return graph


def get_context_files(file_path: str, graph: nx.DiGraph, depth: int = 1) -> List[str]:
    """Return neighbor files up to `depth` import hops from the given file."""
    source = str(Path(file_path).resolve())
    if source not in graph:
        return []

    ego = nx.ego_graph(graph, source, radius=depth, center=False, undirected=False)
    return sorted(ego.nodes())


def _extract_relative_import_paths(content: str) -> Iterable[str]:
    for match in PYTHON_IMPORT_RE.finditer(content):
        import_path = match.group("path")
        if import_path:
            yield import_path

    for match in TS_IMPORT_RE.finditer(content):
        import_path = match.group("path1") or match.group("path2") or match.group("path3")
        if import_path:
            yield import_path


def _resolve_relative_import(source_file: Path, import_path: str) -> Optional[Path]:
    if import_path.startswith("./") or import_path.startswith("../"):
        return _resolve_typescript_relative_import(source_file, import_path)
    if import_path.startswith("."):
        return _resolve_python_relative_import(source_file, import_path)
    return None


def _resolve_typescript_relative_import(source_file: Path, import_path: str) -> Optional[Path]:
    candidate = (source_file.parent / import_path).resolve()
    return _resolve_candidate_path(candidate)


def _resolve_python_relative_import(source_file: Path, import_path: str) -> Optional[Path]:
    leading_dots = len(import_path) - len(import_path.lstrip("."))
    module_suffix = import_path[leading_dots:]
    base_dir = source_file.parent

    for _ in range(max(0, leading_dots - 1)):
        base_dir = base_dir.parent

    if module_suffix:
        candidate = base_dir.joinpath(*module_suffix.split("."))
    else:
        candidate = base_dir

    return _resolve_candidate_path(candidate)


def _resolve_candidate_path(candidate: Path) -> Optional[Path]:
    if candidate.exists():
        if candidate.is_file():
            return candidate.resolve()
        if candidate.is_dir():
            for index_name in VALID_TARGET_INDEX_FILES:
                index_path = candidate / index_name
                if index_path.exists():
                    return index_path.resolve()
            return None

    if candidate.suffix in VALID_SOURCE_EXTENSIONS:
        return candidate.resolve() if candidate.exists() else None

    for suffix in VALID_SOURCE_EXTENSIONS:
        candidate_with_suffix = candidate.with_suffix(suffix)
        if candidate_with_suffix.exists():
            return candidate_with_suffix.resolve()

    for index_name in VALID_TARGET_INDEX_FILES:
        index_path = candidate / index_name
        if index_path.exists():
            return index_path.resolve()

    return None
