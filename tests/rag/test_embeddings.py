import os
import json
import pytest
from app.services.genAI.rag.FAISS.load import RAGManager  # Twój singleton


# Wczytaj ground truth
GT_PATH = os.path.join("tests", "rag", "ground_truth.json")
with open(GT_PATH) as f:
    data = json.load(f)

queries = [q["query"] for q in data["queries"]]
ground_truth_files = [q["files"] for q in data["queries"]]

@pytest.mark.parametrize("query, expected_files", zip(queries, ground_truth_files))
def test_rag_recall(query, expected_files):
    rag = RAGManager()
    docs = rag.vector_store.similarity_search(query, k=10)  # Pobierz top 10 wyników 
    retrieved_files = [doc.metadata['file_path'] for doc in docs]  # Załóżmy, że metadata zawiera nazwę pliku
    print(retrieved_files)
    # sprawdzamy, czy w zwróconym kontekście pojawiają się wszystkie pliki z ground truth
    hits = [f for f in expected_files if f in retrieved_files]  
    recall = len(hits) / len(expected_files)
    
    assert recall >= 0.8, f"Recall dla query '{query}' wynosi {recall:.2f}"

@pytest.mark.parametrize("query, min_files", [
    ("useAuth hook usage", 5),
    ("database connection string", 3),
    ("docker configuration", 2)
])
def test_file_diversity(query, min_files):
    rag = RAGManager()
    all_docs = rag.vector_store.similarity_search(query, k=30)  
    retrieved_files = set(doc.metadata['file_path'] for doc in all_docs) 
    assert len(retrieved_files) >= min_files, f"Zbyt mała różnorodność plików: {len(retrieved_files)}"
