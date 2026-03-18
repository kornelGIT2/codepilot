import ast
import os
import re
from pathlib import PurePosixPath
from typing import Any

skip_list = ["node_modules", ".github", ".venv", "dist", "build", "__pycache__"]
CHUNK_SIZE = 50

TREE_SITTER_LANGUAGE_BY_EXT = {
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}


def _to_posix(path: str) -> str:
    return path.replace("\\", "/")


def _module_key_from_file(file_path: str) -> str:
    module_path = PurePosixPath(_to_posix(file_path))
    without_ext = module_path.with_suffix("")
    parts = list(without_ext.parts)

    if parts and parts[-1] == "index":
        parts = parts[:-1]

    return "/".join(parts)


def _resolve_relative_import(current_file: str, import_value: str) -> str | None:
    if not import_value or not import_value.startswith("."):
        return None

    current_dir = PurePosixPath(_to_posix(current_file)).parent
    raw_target = PurePosixPath(import_value)
    joined = current_dir.joinpath(raw_target)
    normalized = PurePosixPath(os.path.normpath(str(joined)).replace("\\", "/"))

    module_key = _module_key_from_file(str(normalized))
    return module_key.strip("/") or None


def _extract_identifiers_from_text(source: str) -> list[str]:
    keywords = {
        "const", "let", "var", "return", "if", "else", "for", "while", "switch", "case",
        "break", "continue", "function", "class", "import", "from", "export", "default", "new",
        "true", "false", "null", "undefined", "async", "await", "try", "catch", "finally", "this",
    }
    identifiers = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", source)
    filtered = [symbol for symbol in identifiers if symbol not in keywords and len(symbol) > 1]
    return sorted(set(filtered))


def _python_file_symbols(source: str) -> tuple[list[str], list[str], list[str]]:
    defined_symbols: list[str] = []
    referenced_symbols: set[str] = set()
    imports: list[str] = []

    tree = ast.parse(source)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined_symbols.append(node.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            referenced_symbols.add(node.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(("." * node.level) + node.module)
            else:
                imports.append("." * node.level)

    return sorted(set(defined_symbols)), sorted(referenced_symbols), sorted(set(imports))


def _python_ast_chunks(source: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    tree = ast.parse(source)

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        start_line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", None)

        if start_line is None or end_line is None:
            continue

        lines = source.splitlines()
        node_text = "\n".join(lines[start_line - 1:end_line])
        referenced = sorted({
            n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        })

        chunks.append({
            "chunk": node_text,
            "start_line": start_line,
            "end_line": end_line,
            "node_type": type(node).__name__,
            "symbol_name": node.name,
            "defined_symbols": [node.name],
            "referenced_symbols": referenced,
            "chunk_type": "ast_node",
        })

    return chunks


def _load_tree_sitter_parser(ext: str):
    language_name = TREE_SITTER_LANGUAGE_BY_EXT.get(ext)
    if not language_name:
        return None

    try:
        from tree_sitter_languages import get_parser
        return get_parser(language_name)
    except Exception:
        return None


def _tree_sitter_extract_chunks(source: str, ext: str) -> list[dict[str, Any]]:
    parser = _load_tree_sitter_parser(ext)
    if parser is None:
        return []

    language_nodes = {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "lexical_declaration",
        "variable_declaration",
    }

    tree = parser.parse(source.encode("utf-8"))
    root = tree.root_node

    chunks: list[dict[str, Any]] = []

    # We keep extraction at top-level declarations to avoid excessive chunk cardinality.
    for node in root.children:
        target_node = node
        if node.type in {"export_statement", "statement_block"} and node.named_child_count > 0:
            target_node = node.named_children[0]

        if target_node.type not in language_nodes:
            continue

        chunk_text = source.encode("utf-8")[target_node.start_byte:target_node.end_byte].decode("utf-8", errors="ignore")
        start_line = target_node.start_point[0] + 1
        end_line = target_node.end_point[0] + 1
        symbol_candidates = _extract_identifiers_from_text(chunk_text)
        symbol_name = symbol_candidates[0] if symbol_candidates else target_node.type

        chunks.append({
            "chunk": chunk_text,
            "start_line": start_line,
            "end_line": end_line,
            "node_type": target_node.type,
            "symbol_name": symbol_name,
            "defined_symbols": [symbol_name],
            "referenced_symbols": symbol_candidates,
            "chunk_type": "ast_node",
        })

    return chunks


def _line_chunks(source: str) -> list[dict[str, Any]]:
    lines = source.splitlines()
    chunks: list[dict[str, Any]] = []

    for i in range(0, len(lines), CHUNK_SIZE):
        part = lines[i:i + CHUNK_SIZE]
        text = "\n".join(part)
        chunks.append({
            "chunk": text,
            "start_line": i + 1,
            "end_line": min(i + CHUNK_SIZE, len(lines)),
            "node_type": "line_chunk",
            "symbol_name": f"line_chunk_{i + 1}",
            "defined_symbols": [],
            "referenced_symbols": _extract_identifiers_from_text(text),
            "chunk_type": "line_chunk",
        })

    return chunks


def _extract_imports_generic(source: str) -> list[str]:
    patterns = [
        r"import\s+.*?\s+from\s+[\"'](.*?)[\"']",
        r"import\s+[\"'](.*?)[\"']",
        r"require\(\s*[\"'](.*?)[\"']\s*\)",
    ]

    imports: set[str] = set()
    for pattern in patterns:
        for match in re.findall(pattern, source):
            imports.add(match)

    return sorted(imports)


def _build_file_chunks(file_path: str, source: str) -> tuple[list[dict[str, Any]], list[str], list[str], list[str]]:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".py":
        try:
            chunks = _python_ast_chunks(source)
            defined_symbols, referenced_symbols, imports = _python_file_symbols(source)
        except SyntaxError:
            chunks = []
            defined_symbols, referenced_symbols, imports = [], _extract_identifiers_from_text(source), []
    else:
        chunks = _tree_sitter_extract_chunks(source, ext)
        defined_symbols = sorted({c["symbol_name"] for c in chunks if c.get("symbol_name")})
        referenced_symbols = _extract_identifiers_from_text(source)
        imports = _extract_imports_generic(source)

    if not chunks:
        chunks = _line_chunks(source)

    return chunks, defined_symbols, referenced_symbols, imports


def parse_repo(repo_path: str, extensions=None):
    if extensions is None:
        extensions = [".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".html", ".css", ".scss"]

    chunks: list[dict[str, Any]] = []

    for root, _, files in os.walk(repo_path):
        if any(skip in root for skip in skip_list):
            continue

        for file_name in files:
            if not any(file_name.endswith(ext) for ext in extensions):
                continue

            full_path = os.path.join(root, file_name)
            try:
                with open(full_path, "r", encoding="utf-8") as file_handle:
                    source = file_handle.read()
            except UnicodeDecodeError:
                continue

            file_rel_path = _to_posix(os.path.relpath(full_path, repo_path))
            folder_rel = _to_posix(os.path.relpath(root, repo_path))
            module_key = _module_key_from_file(file_rel_path)

            file_chunks, file_defined_symbols, file_referenced_symbols, file_imports = _build_file_chunks(
                file_rel_path,
                source,
            )
            resolved_imports = [
                resolved for resolved in (
                    _resolve_relative_import(file_rel_path, import_path) for import_path in file_imports
                ) if resolved
            ]

            for item in file_chunks:
                chunk_metadata = {
                    "file_path": file_rel_path,
                    "file_name": file_name,
                    "folder": folder_rel,
                    "module_key": module_key,
                    "imports": file_imports,
                    "resolved_imports": resolved_imports,
                    "file_defined_symbols": file_defined_symbols,
                    "file_referenced_symbols": file_referenced_symbols,
                    "defined_symbols": item.get("defined_symbols", []),
                    "referenced_symbols": item.get("referenced_symbols", []),
                    "symbol_name": item.get("symbol_name", ""),
                    "node_type": item.get("node_type", "line_chunk"),
                    "chunk_type": item.get("chunk_type", "line_chunk"),
                    "start_line": item.get("start_line", 1),
                    "end_line": item.get("end_line", 1),
                }
                chunks.append({"metadata": chunk_metadata, "chunk": item["chunk"]})

    return chunks
