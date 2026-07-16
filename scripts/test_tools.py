"""Unit tests for the deterministic tourism tools."""

import unittest

from app.tools import (
    build_itinerary,
    compare_destinations,
    estimate_trip_budget,
    prepare_transport_recommendation,
)


class TourismToolTests(unittest.TestCase):
    """Test valid outputs and rejected tourism-tool inputs."""

    def test_build_itinerary_valid(self) -> None:
        """Build a normalized itinerary request without external calls."""
        result = build_itinerary.invoke(
            {
                "destination": "  Marrakech  ",
                "days": 3,
                "interests": "culture, food",
                "budget_level": "moderate",
            }
        )

        self.assertEqual(result["destination"], "Marrakech")
        self.assertEqual(result["days"], 3)
        self.assertEqual(result["interests"], "culture, food")
        self.assertEqual(result["budget_level"], "moderate")
        self.assertIn("3-day itinerary", result["planning_instructions"])

    def test_compare_destinations_valid(self) -> None:
        """Prepare a criterion-based destination comparison."""
        result = compare_destinations.invoke(
            {
                "destination_a": "Marrakech",
                "destination_b": "Essaouira",
            }
        )

        self.assertEqual(result["destination_a"], "Marrakech")
        self.assertEqual(result["destination_b"], "Essaouira")
        self.assertEqual(
            result["criteria"],
            ["attractions", "climate", "budget", "transport"],
        )
        self.assertIn("criterion by criterion", result["comparison_instructions"])

    def test_estimate_trip_budget_valid(self) -> None:
        """Calculate all daily and trip-level budget totals."""
        result = estimate_trip_budget.invoke(
            {
                "days": 3,
                "accommodation_per_day": 350.25,
                "food_per_day": 150.10,
                "local_transport_per_day": 50,
                "activities_per_day": 100.15,
                "intercity_transport": 300,
            }
        )

        self.assertEqual(result["days"], 3)
        self.assertEqual(result["daily_total_mad"], 650.50)
        self.assertEqual(result["accommodation_total_mad"], 1050.75)
        self.assertEqual(result["food_total_mad"], 450.30)
        self.assertEqual(result["local_transport_total_mad"], 150.00)
        self.assertEqual(result["activities_total_mad"], 300.45)
        self.assertEqual(result["intercity_transport_mad"], 300.00)
        self.assertEqual(result["total_budget_mad"], 2251.50)

    def test_prepare_transport_recommendation_valid(self) -> None:
        """Prepare a transport comparison with stable criteria."""
        result = prepare_transport_recommendation.invoke(
            {
                "origin": "Casablanca",
                "destination": "Fes",
                "preference": "comfort",
            }
        )

        self.assertEqual(result["origin"], "Casablanca")
        self.assertEqual(result["destination"], "Fes")
        self.assertEqual(result["preference"], "comfort")
        self.assertEqual(
            result["comparison_criteria"],
            ["travel time", "estimated cost", "comfort"],
        )
        self.assertIn("comfort preference", result["recommendation_instructions"])

    def test_itinerary_rejects_empty_destination(self) -> None:
        """Reject an itinerary without a destination."""
        with self.assertRaisesRegex(ValueError, "destination"):
            build_itinerary.invoke({"destination": "  ", "days": 2})

    def test_tools_reject_non_positive_days(self) -> None:
        """Reject zero or negative trip durations."""
        with self.assertRaisesRegex(ValueError, "days"):
            build_itinerary.invoke({"destination": "Rabat", "days": 0})
        with self.assertRaisesRegex(ValueError, "days"):
            estimate_trip_budget.invoke(
                {
                    "days": 0,
                    "accommodation_per_day": 100,
                    "food_per_day": 100,
                    "local_transport_per_day": 50,
                    "activities_per_day": 50,
                }
            )
        with self.assertRaisesRegex(ValueError, "days"):
            estimate_trip_budget.invoke(
                {
                    "days": -1,
                    "accommodation_per_day": 100,
                    "food_per_day": 100,
                    "local_transport_per_day": 50,
                    "activities_per_day": 50,
                }
            )

    def test_budget_rejects_negative_price(self) -> None:
        """Reject any negative budget component."""
        with self.assertRaisesRegex(ValueError, "food_per_day"):
            estimate_trip_budget.invoke(
                {
                    "days": 2,
                    "accommodation_per_day": 300,
                    "food_per_day": -1,
                    "local_transport_per_day": 50,
                    "activities_per_day": 100,
                }
            )

    def test_comparison_rejects_identical_destinations(self) -> None:
        """Compare normalized destination values case-insensitively."""
        with self.assertRaisesRegex(ValueError, "must be different"):
            compare_destinations.invoke(
                {
                    "destination_a": "Agadir",
                    "destination_b": "  agadir  ",
                }
            )

    def test_transport_rejects_identical_endpoints(self) -> None:
        """Reject normalized equal transport endpoints."""
        with self.assertRaisesRegex(ValueError, "must differ"):
            prepare_transport_recommendation.invoke(
                {
                    "origin": "Rabat",
                    "destination": " rabat ",
                }
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
