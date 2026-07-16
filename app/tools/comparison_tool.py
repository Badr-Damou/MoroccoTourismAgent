"""Prepare validated inputs for comparing tourism destinations."""

from typing import TypedDict

from app.tools.validators import (
    validate_optional_string,
    validate_required_string,
)


DEFAULT_COMPARISON_CRITERIA = (
    "attractions, climate, budget, transport"
)


class ComparisonRequest(TypedDict):
    """Structured destination-comparison request."""

    destination_a: str
    destination_b: str
    criteria: list[str]
    comparison_instructions: str


def compare_destinations(
    destination_a: str,
    destination_b: str,
    criteria: str = DEFAULT_COMPARISON_CRITERIA,
) -> ComparisonRequest:
    """Validate and structure a destination-comparison request."""
    normalized_destination_a = validate_required_string(
        destination_a,
        "destination_a",
    )
    normalized_destination_b = validate_required_string(
        destination_b,
        "destination_b",
    )
    if normalized_destination_a.casefold() == (
        normalized_destination_b.casefold()
    ):
        raise ValueError("The comparison destinations must be different.")

    normalized_criteria = validate_optional_string(criteria, "criteria")
    criteria_items = [
        item.strip()
        for item in normalized_criteria.split(",")
        if item.strip()
    ]
    criteria_description = (
        ", ".join(criteria_items)
        if criteria_items
        else "the requested travel factors"
    )
    comparison_instructions = (
        f"Compare {normalized_destination_a} and "
        f"{normalized_destination_b} using these criteria: "
        f"{criteria_description}. Present the result criterion by criterion."
    )

    return {
        "destination_a": normalized_destination_a,
        "destination_b": normalized_destination_b,
        "criteria": criteria_items,
        "comparison_instructions": comparison_instructions,
    }
