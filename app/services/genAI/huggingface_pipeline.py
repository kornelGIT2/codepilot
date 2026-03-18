from typing import Generator
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
import torch
from transformers import pipeline, BitsAndBytesConfig
from app.services.genAI.context.utils import get_chat_prompt

DEFAULT_CONTEXT = ""

class TextGenerator:
    def __init__(self):
        self.quant_config = BitsAndBytesConfig(load_in_4bit=True)
        self.pipe = pipeline(
            "text-generation", 
            model="CYFRAGOVPL/Llama-PLLuM-8B-chat",
            dtype=torch.float16,
            model_kwargs={"quantization_config": self.quant_config},
            return_full_text=False,
            temperature=0.5,
            max_new_tokens=512,
        )

        llm = HuggingFacePipeline(pipeline=self.pipe)

        self.chat_model = ChatHuggingFace(llm=llm)


    def generate_stream(self, prompt: str, relevant_chunks: str) -> Generator[str, None, None]:
        selected_txt  = self._classify_intent(prompt)  # Klasyfikacja intencji, można ją wykorzystać do dostosowania promptu lub innych parametrów
        new_template = get_chat_prompt(selected_txt) #temp
        self.chain = new_template | self.chat_model
        for chunk in self.chain.stream({"question": prompt, "context": relevant_chunks}):
            yield chunk.content


    def _classify_intent(self, question: str) -> str: #uzycie mniejszego modelu do klasyfikacji

        classification_template = get_chat_prompt("intent_classification")  # Załaduj szablon klasyfikacji intencji

        classifier_chain = classification_template | self.chat_model        

        response = classifier_chain.invoke({"question": question})
        
        intent = response.content.strip().lower()

        mapping = {
            "wyjaśnij komponent": "explain_component",
            "sprawdź błąd": "bug_check",
            "znajdź zależność": "find_dependency",
            "podsumuj moduł": "summarize_module"
        }

        return mapping.get(intent, "default")