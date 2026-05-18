from langgraph.graph import StateGraph, END
from app.graph.nodes.retrieve import retrieve_node
from app.graph.state import AgentState
from app.graph.edges.conditions import should_continue
from app.providers.huggingface import HuggingFaceProvider
from app.graph.nodes.generate import make_generate_node
from app.graph.nodes.verify import make_verify_node

def build_graph():
    model = HuggingFaceProvider().get_model() 

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", make_generate_node(model))
    workflow.add_node("verify", make_verify_node(model))

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "verify")
    workflow.add_conditional_edges("verify", should_continue, {
        "regenerate": "generate",
        "end": END,
    })

    return workflow.compile()