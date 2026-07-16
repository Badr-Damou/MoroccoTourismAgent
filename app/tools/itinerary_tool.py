"""Prepare validated inputs for itinerary planning."""

from typing import TypedDict

from app.tools.validators import (
    validate_optional_string,
    validate_positive_integer,
    validate_required_string,
)


class ItineraryRequest(TypedDict):
    """Structured itinerary-planning request."""

    destination: str
    days: int
    interests: str
    budget_level: str
    planning_instructions: str


def build_itinerary(
    destination: str,
    days: int,
    interests: str = "",
    budget_level: str = "moderate",
) -> ItineraryRequest:
    """Validate and structure an itinerary-planning request."""
    normalized_destination = validate_required_string(
        destination,
        "destination",
    )
    normalized_days = validate_positive_integer(days, "days")
    normalized_interests = validate_optional_string(interests, "interests")
    normalized_budget = validate_optional_string(
        budget_level,
        "budget_level",
    )
    if not normalized_budget:
        normalized_budget = "moderate"

    interest_instruction = (
        f" Prioritize these interests: {normalized_interests}."
        if normalized_interests
        else " Balance cultural, practical, and leisure activities."
    )
    planning_instructions = (
        f"Prepare a {normalized_days}-day itinerary for "
        f"{normalized_destination}, organized by day, for a "
        f"{normalized_budget} budget.{interest_instruction}"
    )

    return {
        "destination": normalized_destination,
        "days": normalized_days,
        "interests": normalized_interests,
        "budget_level": normalized_budget,
        "planning_instructions": planning_instructions,
    }
