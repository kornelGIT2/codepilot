from app.rag.faiss.load import RAGManager
from app.core.config import settings

_rag_manager = RAGManager()

def retrieve_context(question: str) -> str:
    data = _rag_manager().get_context(question, k=settings.top_k)
    return "\n\n".join([chunk["content"] for chunk in data])