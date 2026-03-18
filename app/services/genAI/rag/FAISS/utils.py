from langchain_core.embeddings import Embeddings


class JinaLangChainAdapter(Embeddings):
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedding_model.encode(texts=texts, task="retrieval", prompt_name="document").detach().float().cpu().numpy().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embedding_model.encode(texts=[text], task="retrieval", prompt_name="query").detach().float().cpu().numpy().tolist()[0]