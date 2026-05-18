import json
import os
from typing import Any

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from app.rag.embeddings import EmbeddingModel
from app.rag.parser import parse_repo
from app.rag.faiss.utils import JinaLangChainAdapter
from app.rag.utils import normalize_l2

GT_PATH = os.path.join("tests", "rag", "ground_truth.json")
REPO_PATH = os.path.join("D:/CodePilot/repos/react-practices")


def normalize_expected_paths(paths: list[str]) -> list[str]:
    return [p.replace('\\', '/').strip('/') for p in paths]


def build_vector_store(use_tree_sitter: bool) -> FAISS:
    chunks = parse_repo(REPO_PATH, use_tree_sitter=use_tree_sitter)
    texts = [chunk["chunk"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    model = EmbeddingModel()
    adapter = JinaLangChainAdapter(model)

    batch_size = 32
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(texts=batch, task="retrieval", prompt_name="document")
        emb_np = emb.detach().float().cpu().numpy()
        emb_norm = normalize_l2(emb_np)
        embeddings.append(emb_norm)

    embeddings = np.vstack(embeddings)
    text_embeddings = [(text, emb.tolist()) for text, emb in zip(texts, embeddings)]

    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=adapter,
        metadatas=metadatas,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    return vector_store


def evaluate(vector_store: FAISS, queries: list[str], expected_files: list[list[str]], k: int = 10) -> dict[str, Any]:
    results = []
    for query, expected in zip(queries, expected_files):
        docs = vector_store.similarity_search(query, k=k)
        retrieved = [doc.metadata.get("file_path", "") for doc in docs]
        expected_norm = normalize_expected_paths(expected)
        hits = [f for f in expected_norm if f in retrieved]
        recall = len(hits) / len(expected_norm) if expected_norm else 0.0
        precision = len(hits) / len(retrieved) if retrieved else 0.0
        results.append({
            "query": query,
            "expected": expected_norm,
            "retrieved": retrieved,
            "hits": hits,
            "recall": recall,
            "precision": precision,
        })

    average_recall = sum(r["recall"] for r in results) / len(results)
    average_precision = sum(r["precision"] for r in results) / len(results)

    return {
        "results": results,
        "average_recall": average_recall,
        "average_precision": average_precision,
    }


def load_ground_truth():
    with open(GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)
    queries = [q["query"] for q in gt["queries"]]
    expected_files = [q["files"] for q in gt["queries"]]
    return queries, expected_files


if __name__ == "__main__":
    queries, expected_files = load_ground_truth()

    print("Building line-only index...")
    line_index = build_vector_store(use_tree_sitter=False)
    print("Evaluating line-only index...")
    line_metrics = evaluate(line_index, queries, expected_files)

    print("Building AST index...")
    ast_index = build_vector_store(use_tree_sitter=True)
    print("Evaluating AST index...")
    ast_metrics = evaluate(ast_index, queries, expected_files)

    print("\n=== Comparison ===")
    print(f"Line-only chunks: {len(line_index.docstore._dict) if getattr(line_index, 'docstore', None) else 'unknown'} documents")
    print(f"AST-enabled chunks: {len(ast_index.docstore._dict) if getattr(ast_index, 'docstore', None) else 'unknown'} documents")
    print(f"Line-only average recall: {line_metrics['average_recall']:.3f}")
    print(f"AST-enabled average recall: {ast_metrics['average_recall']:.3f}")
    print(f"Line-only average precision: {line_metrics['average_precision']:.3f}")
    print(f"AST-enabled average precision: {ast_metrics['average_precision']:.3f}")
    print(f"Precision increase: {ast_metrics['average_precision'] - line_metrics['average_precision']:.3f}")
    print(f"Recall increase: {ast_metrics['average_recall'] - line_metrics['average_recall']:.3f}")
