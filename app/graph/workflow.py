"""Assemble and compile the manual LangGraph tourism workflow."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.edges import route_after_validation, route_by_intent
from app.graph.nodes import (
    classify_intent_node,
    compare_destinations_node,
    estimate_budget_node,
    generate_answer_node,
    plan_itinerary_node,
    retrieve_node,
    transport_recommendation_node,
    validate_answer_node,
)
from app.graph.state import TourismAgentState


def build_graph() -> CompiledStateGraph:
    """Build the tourism workflow with isolated in-memory thread history."""
    workflow = StateGraph(TourismAgentState)

    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("plan_itinerary", plan_itinerary_node)
    workflow.add_node("compare_destinations", compare_destinations_node)
    workflow.add_node("estimate_budget", estimate_budget_node)
    workflow.add_node(
        "transport_recommendation",
        transport_recommendation_node,
    )
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("validate_answer", validate_answer_node)

    workflow.add_edge(START, "classify_intent")
    workflow.add_edge("classify_intent", "retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        route_by_intent,
        {
            "factual": "generate_answer",
            "general": "generate_answer",
            "itinerary": "plan_itinerary",
            "comparison": "compare_destinations",
            "budget": "estimate_budget",
            "transport": "transport_recommendation",
        },
    )
    workflow.add_edge("plan_itinerary", "generate_answer")
    workflow.add_edge("compare_destinations", "generate_answer")
    workflow.add_edge("estimate_budget", "generate_answer")
    workflow.add_edge("transport_recommendation", "generate_answer")
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
