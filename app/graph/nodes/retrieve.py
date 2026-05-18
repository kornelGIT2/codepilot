from app.tools.retriever import retrieve_context
from app.graph.state import AgentState

def retrieve_node(state: AgentState) -> AgentState:
    context = retrieve_context(state["question"])
    return {"context": context}