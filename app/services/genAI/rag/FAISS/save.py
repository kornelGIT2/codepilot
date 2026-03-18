import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_community.vectorstores.utils import DistanceStrategy
from app.services.genAI.rag.embeddings import embeddings_pipeline
from langchain_community.vectorstores import FAISS
from FAISS.utils import JinaLangChainAdapter
import numpy as np


# PersistentClient działa w v1# embedding_function=None, bo mamy już gotowe embeddingi
all_embeddings, all_metadatas, model_instance = embeddings_pipeline()

adapter = JinaLangChainAdapter(model_instance)

text_embeddings = [(meta['text'], emb) for meta, emb in zip(all_metadatas, all_embeddings.tolist())]

# Tworzymy bazę z GOTOWYCH danych
vector_store = FAISS.from_embeddings(
    text_embeddings=text_embeddings,
    embedding=adapter,
    metadatas=all_metadatas,
    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
)

base_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(base_dir, "faiss_index")

vector_store.save_local(DB_PATH)