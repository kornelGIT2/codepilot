from types import SimpleNamespace

from langchain_core.documents import Document

from app.services.genAI.rag.FAISS.load import RAGManager


class _FakeDocStore:
    def __init__(self, docs):
        self._dict = docs


class _FakeVectorStore:
    def __init__(self, docs):
        self.docstore = _FakeDocStore(docs)


def test_cross_reference_retrieval_by_symbol_and_module():
    manager = object.__new__(RAGManager)

    docs = {
        "1": Document(
            page_content="def normalize_name(value): return value.strip().lower()",
            metadata={
                "file_path": "pkg/utils.py",
                "start_line": 1,
                "end_line": 2,
                "symbol_name": "normalize_name",
                "defined_symbols": ["normalize_name"],
                "module_key": "pkg/utils",
            },
        ),
        "2": Document(
            page_content="class MetricsService: pass",
            metadata={
                "file_path": "pkg/metrics.py",
                "start_line": 1,
                "end_line": 1,
                "symbol_name": "MetricsService",
                "defined_symbols": ["MetricsService"],
                "module_key": "pkg/metrics",
            },
        ),
    }
    manager.vector_store = _FakeVectorStore(docs)

    base_results = [
        {
            "content": "from .utils import normalize_name\nvalue = normalize_name(name)",
            "metadata": {
                "file_path": "pkg/service.py",
                "start_line": 1,
                "end_line": 2,
                "symbol_name": "process",
                "referenced_symbols": ["normalize_name"],
                "resolved_imports": ["pkg/utils"],
            },
            "score": 0.95,
        }
    ]

    cross_refs = manager._find_cross_references(base_results, limit=3)

    assert cross_refs
    assert cross_refs[0]["metadata"]["file_path"] == "pkg/utils.py"
    assert "cross_ref_relation" in cross_refs[0]["metadata"]
