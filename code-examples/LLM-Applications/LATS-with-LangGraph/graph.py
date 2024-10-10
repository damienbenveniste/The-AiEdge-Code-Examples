from langgraph.graph import END, StateGraph, START
from nodes import TreeState, initialize, expand, should_continue, get_best_solution


builder = StateGraph(TreeState)
builder.add_node("initialize", initialize)
builder.add_node("expand", expand)
builder.add_node("get_best_solution", get_best_solution)

builder.add_edge(START, "initialize")
builder.add_edge("initialize", "expand")
builder.add_conditional_edges(
    "expand",
    # Either continue to rollout or finish
    should_continue,
    {
        "expand": "expand",
        "terminate": 'get_best_solution',
    },
)
builder.add_edge("get_best_solution", END)
graph = builder.compile()