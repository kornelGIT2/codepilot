from app.graph.graph import build_graph

graph = build_graph()

print(graph.get_graph().draw_mermaid())