import os
from langchain_core.prompts import ChatPromptTemplate


PROMPT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

def load_prompt_raw(prompt_name: str) -> str:
    prompt_file = prompt_name if prompt_name.endswith(".txt") else f"{prompt_name}.txt"
    path = os.path.join(PROMPT_FOLDER, prompt_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Brak pliku promptu: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def get_chat_prompt(name: str) -> ChatPromptTemplate:
    raw_content = load_prompt_raw(name)
    
  
    if "---" in raw_content:
        parts = raw_content.split("---", 1)
        system_part = parts[0].strip()
        user_part = parts[1].strip()
        
        return ChatPromptTemplate.from_messages([
            ("system", system_part),
            ("user", user_part),
        ])
    
   
    return ChatPromptTemplate.from_messages([
        ("user", raw_content.strip()),
    ])
