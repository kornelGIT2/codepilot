from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
import torch
from transformers import pipeline, BitsAndBytesConfig
from app.core.config import settings


class HuggingFaceProvider:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        quant_config = BitsAndBytesConfig(load_in_4bit=True)
        pipe = pipeline(
            "text-generation",
            model=settings.model_name,
            dtype=torch.float16,
            model_kwargs={"quantization_config": quant_config},
            return_full_text=False,
            temperature=settings.temperature,
            max_new_tokens=settings.max_new_tokens,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        self._model = ChatHuggingFace(llm=llm)
        self._initialized = True


    def get_model(self): 
        return self._model        