import sys
import os
from copy import deepcopy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_community.vectorstores import FAISS
from FAISS.utils import JinaLangChainAdapter
from app.services.genAI.rag.models.jina_embedding import EmbeddingModel
from langchain_community.vectorstores.utils import DistanceStrategy
from app.services.genAI.rag.utils import normalize_l2

class RAGManager:
    _instance = None

    def __new__(cls):
        # Prosty wzorzec Singleton - gwarantuje tylko jedną instancję w aplikacji
        if cls._instance is None:
            cls._instance = super(RAGManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(base_dir, "faiss_index")
        
        self.emb_model = EmbeddingModel()
        self.adapter = JinaLangChainAdapter(self.emb_model)
        
        self.vector_store = FAISS.load_local(
            self.db_path, 
            self.adapter, 
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
        )
        self._initialized = True

    def _all_documents(self):
        docstore = getattr(self.vector_store, "docstore", None)
        docs_map = getattr(docstore, "_dict", None)

        if isinstance(docs_map, dict):
            return list(docs_map.values())
        return []

    def _find_cross_references(self, base_results: list[dict], limit: int) -> list[dict]:
        if limit <= 0:
            return []

        needed_symbols = set()
        needed_modules = set()
        seen_keys = set()

        for item in base_results:
            metadata = item.get("metadata", {})
            needed_symbols.update(metadata.get("referenced_symbols", []))
            needed_modules.update(metadata.get("resolved_imports", []))

            unique_key = (
                metadata.get("file_path", ""),
                metadata.get("start_line", 0),
                metadata.get("symbol_name", ""),
            )
            seen_keys.add(unique_key)

        candidates = []
        for doc in self._all_documents():
            metadata = doc.metadata or {}
            unique_key = (
                metadata.get("file_path", ""),
                metadata.get("start_line", 0),
                metadata.get("symbol_name", ""),
            )

            if unique_key in seen_keys:
                continue

            symbol_defs = set(metadata.get("defined_symbols", []))
            module_key = metadata.get("module_key")

            relation_hits = []
            if symbol_defs.intersection(needed_symbols):
                relation_hits.append("symbol_reference")
            if module_key and module_key in needed_modules:
                relation_hits.append("module_import")

            if not relation_hits:
                continue

            relation_weight = 0.90 if "symbol_reference" in relation_hits else 0.84
            enriched_metadata = deepcopy(metadata)
            enriched_metadata["cross_ref_relation"] = relation_hits

            candidates.append({
                "content": doc.page_content,
                "metadata": enriched_metadata,
                "score": relation_weight,
            })

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:limit]

    def get_context(self, query: str, k: int, threshold: float = 0.8, cross_ref_k: int = 3) -> list[dict]:

        emb_query = self.emb_model.encode(texts=[query], task="retrieval", prompt_name="query").detach().float().cpu().numpy()
        normalized_emb_query = normalize_l2(emb_query)

        print(f"shape of normalized_emb_query: {normalized_emb_query.shape}")

        docs = self.vector_store.similarity_search_with_score_by_vector(normalized_emb_query[0], k=k)

        filtered_data = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            }
            for doc, score in docs
            if float(score) >= threshold
        ]

        if not filtered_data:
            return []  # Brak wystarczająco podobnych dokumentów

        # Expand context with cross-module definitions and dependency targets.
        cross_refs = self._find_cross_references(filtered_data, limit=cross_ref_k)

        if not cross_refs:
            return filtered_data

        combined = filtered_data + cross_refs

        # Deduplicate entries by file and line span, preserving highest score first.
        unique_by_key = {}
        for item in sorted(combined, key=lambda row: row["score"], reverse=True):
            metadata = item.get("metadata", {})
            key = (
                metadata.get("file_path", ""),
                metadata.get("start_line", 0),
                metadata.get("end_line", 0),
                metadata.get("symbol_name", ""),
            )
            if key not in unique_by_key:
                unique_by_key[key] = item

        final_results = list(unique_by_key.values())
        final_results.sort(key=lambda item: item["score"], reverse=True)

        return final_results[:k + cross_ref_k]

    def health_check(self) -> bool:
        try:
            test_query = "Testowy zapytanie do sprawdzenia RAG"
            self.get_context(test_query, k=1)
            return True
        except Exception as e:
            print(f"RAGManager health check failed: {e}")
            return False
        