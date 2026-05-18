from pathlib import Path

import networkx as nx

from app.graph.dependency import build_dependency_graph, get_context_files, parse_imports


def test_parse_imports_python_relative(tmp_path: Path):
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()

    service = pkg_dir / "service.py"
    service.write_text("from .utils import normalize_name\n", encoding="utf-8")

    utils = pkg_dir / "utils.py"
    utils.write_text("def normalize_name(value: str) -> str:\n    return value.strip().lower()\n", encoding="utf-8")

    imports = parse_imports(str(service), service.read_text(encoding="utf-8"))

    assert len(imports) == 1
    assert imports[0].endswith(str(utils.resolve()))


def test_parse_imports_typescript_relative(tmp_path: Path):
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()

    jwt = lib_dir / "jwt.ts"
    jwt.write_text("export const token = 'secret'\n", encoding="utf-8")

    auth = lib_dir / "auth.ts"
    auth.write_text("import { token } from './jwt'\n", encoding="utf-8")

    imports = parse_imports(str(auth), auth.read_text(encoding="utf-8"))

    assert len(imports) == 1
    assert imports[0].endswith(str(jwt.resolve()))


def test_build_dependency_graph_and_context(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    main = repo_root / "main.py"
    helper = repo_root / "helper.py"
    util = repo_root / "util.py"

    helper.write_text("from .util import helper_value\n", encoding="utf-8")
    main.write_text("from .helper import helper_value\n", encoding="utf-8")
    util.write_text("HELPER_VALUE = 42\n", encoding="utf-8")

    graph = build_dependency_graph(str(repo_root))

    assert isinstance(graph, nx.DiGraph)
    assert graph.has_edge(str(main.resolve()), str(helper.resolve()))
    assert graph.has_edge(str(helper.resolve()), str(util.resolve()))
    assert not graph.has_edge(str(main.resolve()), str(util.resolve()))

    direct_context = get_context_files(str(main.resolve()), graph, depth=1)
    assert direct_context == [str(helper.resolve())]

    two_hop_context = get_context_files(str(main.resolve()), graph, depth=2)
    assert set(two_hop_context) == {str(helper.resolve()), str(util.resolve())}
