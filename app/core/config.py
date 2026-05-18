# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # model
    model_name: str = "CYFRAGOVPL/Llama-PLLuM-8B-chat"
    temperature: float = 0.3
    max_new_tokens: int = 512

    # graph
    max_iterations: int = 3

    # rag
    faiss_index_path: str = "app/rag/faiss/store"
    top_k: int = 5

    # api
    allowed_origins: list[str] = ["*"]

    class Config:
        env_file = ".env"

settings = Settings()