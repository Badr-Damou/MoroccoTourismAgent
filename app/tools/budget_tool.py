"""Calculate a deterministic trip-budget estimate."""

from typing import TypedDict

from app.tools.validators import (
    validate_non_negative_number,
    validate_positive_integer,
)


class TripBudgetEstimate(TypedDict):
    """Calculated trip-budget totals in Moroccan dirhams."""

    days: int
    daily_total_mad: float
    accommodation_total_mad: float
    food_total_mad: float
    local_transport_total_mad: float
    activities_total_mad: float
    intercity_transport_mad: float
    total_budget_mad: float


def estimate_trip_budget(
    days: int,
    accommodation_per_day: float,
    food_per_day: float,
    local_transport_per_day: float,
    activities_per_day: float,
    intercity_transport: float = 0,
) -> TripBudgetEstimate:
    """Validate costs and calculate rounded trip totals in MAD."""
    normalized_days = validate_positive_integer(days, "days")
    accommodation = validate_non_negative_number(
        accommodation_per_day,
        "accommodation_per_day",
    )
    food = validate_non_negative_number(food_per_day, "food_per_day")
    local_transport = validate_non_negative_number(
        local_transport_per_day,
        "local_transport_per_day",
    )
    activities = validate_non_negative_number(
        activities_per_day,
        "activities_per_day",
    )
    intercity = validate_non_negative_number(
        intercity_transport,
        "intercity_transport",
    )

    daily_total = accommodation + food + local_transport + activities
    accommodation_total = accommodation * normalized_days
    food_total = food * normalized_days
    local_transport_total = local_transport * normalized_days
    activities_total = activities * normalized_days
    total_budget = daily_total * normalized_days + intercity

    return {
        "days": normalized_days,
        "daily_total_mad": round(daily_total, 2),
        "accommodation_total_mad": round(accommodation_total, 2),
        "food_total_mad": round(food_total, 2),
        "local_transport_total_mad": round(local_transport_total, 2),
        "activities_total_mad": round(activities_total, 2),
        "intercity_transport_mad": round(intercity, 2),
        "total_budget_mad": round(total_budget, 2),
    }
