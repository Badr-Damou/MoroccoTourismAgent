"""Define conditional routing rules for the tourism-agent graph."""

from typing import Literal

from app.graph.state import TourismAgentState


IntentRoute = Literal[
    "factual",
    "general",
    "itinerary",
    "comparison",
    "budget",
    "transport",
]


def route_by_intent(state: TourismAgentState) -> IntentRoute:
    """Route deterministically from the classified tourism intent."""
    intent = state.get("intent", "general")
    if intent == "factual":
        return "factual"
    if intent == "itinerary":
        return "itinerary"
    if intent == "comparison":
        return "comparison"
    if intent == "budget":
        return "budget"
    if intent == "transport":
        return "transport"
    return "general"


def route_after_validation(
    state: TourismAgentState,
) -> Literal["end", "revise"]:
    """End successful/unsupported runs or allow one answer revision."""
    validation_result = state.get("validation_result", "needs_revision")
    revision_count = state.get("revision_count", 0)

    if validation_result in {"valid", "insufficient_information"}:
        return "end"
    if validation_result == "needs_revision" and revision_count < 1:
        return "revise"
    return "end"
