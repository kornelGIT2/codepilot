from app.graph.state import AgentState
from app.core.config import settings

def should_continue(state: AgentState) -> str:
    if state["iterations"] >= settings.max_iterations:
        return "end"
    if "APPROVED" in state["feedback"]:
        return "end"
    return "regenerate"