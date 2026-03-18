import numpy as np

def cosine_similarity(vecA: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """
    vecA: (D,)
    embeddings: (N, D)
    returns: (N,) cosine similarities
    """
    print(embeddings.shape, vecA.shape)
    vecA = np.asarray(vecA)
    dot_product = embeddings @ vecA  # (N,)
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(embeddings, axis=1)  # (N,)

    # unikamy dzielenia przez 0
    normB_safe = np.where(normB == 0, 1e-10, normB)
    normA_safe = normA if normA != 0 else 1e-10

    similarities = dot_product / (normA_safe * normB_safe)
    return similarities

def normalize_l2(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)