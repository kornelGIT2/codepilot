from transformers import AutoModel
import torch

class EmbeddingModel:
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v5-text-small",
            trust_remote_code=True,
            dtype=torch.bfloat16,  # Recommended for GPUs
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device=device)

# Optional: set truncate_dim and max_length in encode() to control embedding size and input length


    def encode(self, texts, task="retrieval", prompt_name=None):
        return self.model.encode(
                texts=texts,
                task=task,
                prompt_name=prompt_name,
        )