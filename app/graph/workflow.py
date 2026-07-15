"""Assemble and compile the manual LangGraph tourism workflow."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.edges import route_after_validation
from app.graph.nodes import (
    classify_intent_node,
    generate_answer_node,
    retrieve_node,
    validate_answer_node,
)
from app.graph.state import TourismAgentState


def build_graph() -> CompiledStateGraph:
    """Build the tourism workflow with isolated in-memory thread history."""
    workflow = StateGraph(TourismAgentState)

    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("validate_answer", validate_answer_node)

    workflow.add_edge(START, "classify_intent")
    workflow.add_edge("classify_intent", "retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", "validate_answer")
    workflow.add_conditional_edges(
        "validate_answer",
        route_after_validation,
        {
            "end": END,
            "revise": "generate_answer",
        },
    )

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
