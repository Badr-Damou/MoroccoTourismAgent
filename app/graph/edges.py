"""Define conditional routing rules for the tourism-agent graph."""

from typing import Literal

from app.graph.state import TourismAgentState


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
