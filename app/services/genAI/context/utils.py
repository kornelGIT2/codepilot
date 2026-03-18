import os
from langchain_core.prompts import ChatPromptTemplate

# Zakładając, że plik jest w app/utils/prompts.py, 
# wychodzimy dwa poziomy wyżej do głównego katalogu prompts
PROMPT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

def load_prompt_raw(prompt_name: str) -> str:
    path = os.path.join(PROMPT_FOLDER, f"{prompt_name}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Brak pliku promptu: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def get_chat_prompt(name: str) -> ChatPromptTemplate:
    raw_content = load_prompt_raw(name)
    
    # Obsługa formatu System --- User
    if "---" in raw_content:
        parts = raw_content.split("---", 1)
        system_part = parts[0].strip()
        user_part = parts[1].strip()
        
        return ChatPromptTemplate.from_messages([
            ("system", system_part),
            ("user", user_part),
        ])
    
    # Fallback: jeśli brak separatora, traktuj wszystko jako wiadomość użytkownika
    return ChatPromptTemplate.from_messages([
        ("user", raw_content.strip()),
    ])
