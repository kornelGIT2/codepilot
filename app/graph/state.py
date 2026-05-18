from typing import TypedDict

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    feedback: str
    iterations: int