import sys
import os
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

    def get_context(self, query: str, k: int, threshold: float = 0.8) -> list[dict]:

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
        

        return filtered_data

    def health_check(self) -> bool:
        try:
            test_query = "Testowy zapytanie do sprawdzenia RAG"
            self.get_context(test_query, k=1)
            return True
        except Exception as e:
            print(f"RAGManager health check failed: {e}")
            return False
        