from app.services.genAI.rag.models.jina_embedding import EmbeddingModel
from app.services.genAI.rag.parser import parse_repo
import os
import numpy as np
from app.services.genAI.rag.utils import normalize_l2


def embeddings_pipeline():

    embeddingModel = EmbeddingModel()
    BATCH_SIZE = 32

    repo_path = os.path.join("D:/CodePilot/repos/react-practices")
    chunks = parse_repo(repo_path)

    all_embeddings = [] 
    all_metadatas = []  


    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i+BATCH_SIZE]
        texts = [chunk['chunk'] for chunk in batch_chunks]

        embeddings = embeddingModel.encode(texts=texts, task="retrieval", prompt_name="document")

        embeddings_np = embeddings.detach().float().cpu().numpy()  
        embeddings_normalized = normalize_l2(embeddings_np)

        all_embeddings.append(embeddings_normalized) 

        for chunk in batch_chunks:
            meta = chunk['metadata'].copy()
            meta['text'] = chunk['chunk'] 
            all_metadatas.append(meta)

    all_embeddings = np.vstack(all_embeddings) 

    return all_embeddings, all_metadatas, embeddingModel


