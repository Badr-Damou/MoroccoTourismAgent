"""Prepare validated inputs for transport recommendations."""

from typing import TypedDict

from app.tools.validators import (
    validate_optional_string,
    validate_required_string,
)


TRANSPORT_COMPARISON_CRITERIA = [
    "travel time",
    "estimated cost",
    "comfort",
]


class TransportRecommendationRequest(TypedDict):
    """Structured transport-recommendation request."""

    origin: str
    destination: str
    preference: str
    comparison_criteria: list[str]
    recommendation_instructions: str


def prepare_transport_recommendation(
    origin: str,
    destination: str,
    preference: str = "balanced",
) -> TransportRecommendationRequest:
    """Validate and structure a transport-recommendation request."""
    normalized_origin = validate_required_string(origin, "origin")
    normalized_destination = validate_required_string(
        destination,
        "destination",
    )
    if normalized_origin.casefold() == normalized_destination.casefold():
        raise ValueError("The transport origin and destination must differ.")

    normalized_preference = validate_optional_string(
        preference,
        "preference",
    )
    if not normalized_preference:
        normalized_preference = "balanced"

    criteria = TRANSPORT_COMPARISON_CRITERIA.copy()
    recommendation_instructions = (
        f"Compare transport options from {normalized_origin} to "
        f"{normalized_destination} by {', '.join(criteria)}. Recommend the "
        f"option that best matches a {normalized_preference} preference."
    )

    return {
        "origin": normalized_origin,
        "destination": normalized_destination,
        "preference": normalized_preference,
        "comparison_criteria": criteria,
        "recommendation_instructions": recommendation_instructions,
    }
