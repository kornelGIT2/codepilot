from app.prompts.loader import get_chat_prompt
from app.graph.state import AgentState

def make_verify_node(model):
    def verify_node(state: AgentState) -> AgentState:
        chain = get_chat_prompt("verify") | model
        result = chain.invoke({
            "question": state["question"],
            "answer": state["answer"],
            "context": state["context"],
        })
        return {"feedback": result.content}
    return verify_node