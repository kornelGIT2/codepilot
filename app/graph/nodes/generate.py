from app.prompts.loader import get_chat_prompt
from app.graph.state import AgentState

def make_generate_node(model):
    def generate_node(state: AgentState) -> AgentState:
        chain = get_chat_prompt("generate") | model
        full_response = ""
        for chunk in chain.stream({
            "question": state["question"],
            "context": state["context"],
            "feedback": state.get("feedback", ""),
        }):
            full_response += chunk.content
        return {"answer": full_response, "iterations": state.get("iterations", 0) + 1}
    return generate_node