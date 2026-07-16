"""Test deterministic graph decisions without external API requests."""

import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.graph.edges import route_by_intent
from app.graph.nodes import (
    _invoke_gemini,
    classify_intent_node,
    classify_tourism_intent,
    compare_destinations_node,
    estimate_budget_node,
    generate_answer_node,
    plan_itinerary_node,
    transport_recommendation_node,
    validate_answer_node,
)


class FakeQuotaError(RuntimeError):
    """Represent a Gemini quota response without making an API request."""

    code = 429


class FakeTemporaryError(RuntimeError):
    """Represent a retryable Gemini service response."""

    code = 503


class FakeClientError(RuntimeError):
    """Represent a non-retryable Gemini request response."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


class GraphNodeTests(unittest.TestCase):
    """Cover quota-saving classification and validation behavior."""

    def test_deterministic_intent_rules(self) -> None:
        """Classify representative questions into all supported intents."""
        examples = {
            "What are the main attractions in Marrakech?": "factual",
            "Plan a two-day itinerary for Chefchaouen.": "itinerary",
            "Compare Marrakech versus Essaouira.": "comparison",
            (
                "Estimate the budget for three days in Marrakech for a "
                "moderate traveler."
            ): "budget",
            "How can I travel from Fez to Rabat by train?": "transport",
            "Which city should I visit?": "general",
        }

        for question, expected_intent in examples.items():
            with self.subTest(question=question):
                self.assertEqual(
                    classify_tourism_intent(question),
                    expected_intent,
                )

    @patch("app.graph.nodes._invoke_gemini")
    def test_classification_node_does_not_call_gemini(
        self,
        invoke_gemini,
    ) -> None:
        """Keep the existing graph node contract without using the LLM."""
        result = classify_intent_node(
            {"question": "Plan a day trip to Marrakech."}
        )

        self.assertEqual(result["intent"], "itinerary")
        invoke_gemini.assert_not_called()

    def test_route_by_intent_supports_every_path(self) -> None:
        """Route every supported intent without an LLM call."""
        for intent in (
            "factual",
            "general",
            "itinerary",
            "comparison",
            "budget",
            "transport",
        ):
            with self.subTest(intent=intent):
                self.assertEqual(route_by_intent({"intent": intent}), intent)
        self.assertEqual(route_by_intent({"intent": "unknown"}), "general")

    def test_specialized_nodes_prepare_expected_tool_results(self) -> None:
        """Prepare itinerary, comparison, and transport inputs locally."""
        itinerary = plan_itinerary_node(
            {
                "question": "Plan a two-day trip to Chefchaouen.",
                "user_preferences": ["quiet", "moderate budget"],
            }
        )
        comparison = compare_destinations_node(
            {
                "question": "Compare Marrakech and Chefchaouen.",
                "user_preferences": ["cultural"],
            }
        )
        transport = transport_recommendation_node(
            {
                "question": (
                    "What is the fastest way to travel from Tangier to "
                    "Chefchaouen?"
                ),
            }
        )

        self.assertEqual(itinerary["selected_path"], "itinerary")
        self.assertEqual(itinerary["itinerary_result"]["days"], 2)
        self.assertEqual(comparison["selected_path"], "comparison")
        self.assertEqual(
            comparison["comparison_result"]["destination_b"],
            "Chefchaouen",
        )
        self.assertEqual(transport["selected_path"], "transport")
        self.assertEqual(
            transport["transport_result"]["preference"],
            "fastest",
        )

    def test_budget_node_scopes_prices_to_requested_destination(self) -> None:
        """Never mix another destination's prices into a budget result."""
        context = """[Source 1: Chefchaoun.pdf, page 5]
Mid-range Traveler Accommodation: 700 MAD Food: 250 MAD
Transportation: 150 MAD Activities: 200 MAD Estimated Total: 1,300 MAD/day

[Source 2: Marrakech.pdf, page 7]
Mid-range Traveler Accommodation: 900 MAD Food: 350 MAD
Transportation: 150 MAD Activities: 350 MAD Estimated Total: 1,750 MAD/day
"""

        result = estimate_budget_node(
            {
                "question": (
                    "Estimate the budget for three days in Marrakech for a "
                    "moderate traveler."
                ),
                "context": context,
            }
        )

        self.assertEqual(result["selected_path"], "budget")
        self.assertEqual(
            result["budget_result"],
            {
                "days": 3,
                "daily_total_mad": 1750.0,
                "accommodation_total_mad": 2700.0,
                "food_total_mad": 1050.0,
                "local_transport_total_mad": 450.0,
                "activities_total_mad": 1050.0,
                "intercity_transport_mad": 0.0,
                "total_budget_mad": 5250.0,
            },
        )

    @patch("app.graph.nodes._invoke_chat_model")
    def test_budget_answer_uses_tool_values_verbatim(self, invoke_model) -> None:
        """Render every three-day Marrakech total without LLM arithmetic."""
        result = generate_answer_node(
            {
                "question": (
                    "Estimate the budget for three days in Marrakech for a "
                    "moderate traveler."
                ),
                "intent": "budget",
                "selected_path": "budget",
                "retrieved_documents": [
                    Document(
                        page_content="Moderate prices.",
                        metadata={"filename": "Marrakech.pdf", "page": 7},
                    )
                ],
                "context": "[Source 1: Marrakech.pdf, page 7] Moderate prices.",
                "budget_result": {
                    "days": 3,
                    "daily_total_mad": 1750.0,
                    "accommodation_total_mad": 2700.0,
                    "food_total_mad": 1050.0,
                    "local_transport_total_mad": 450.0,
                    "activities_total_mad": 1050.0,
                    "intercity_transport_mad": 0.0,
                    "total_budget_mad": 5250.0,
                },
                "revision_count": 0,
            }
        )

        invoke_model.assert_not_called()
        self.assertIn("Accommodation: 2,700.00 MAD", result["final_answer"])
        self.assertIn("Food: 1,050.00 MAD", result["final_answer"])
        self.assertIn("Local transport: 450.00 MAD", result["final_answer"])
        self.assertIn("Activities: 1,050.00 MAD", result["final_answer"])
        self.assertIn("Intercity transport: 0.00 MAD", result["final_answer"])
        self.assertIn("Daily total: 1,750.00 MAD", result["final_answer"])
        self.assertIn("Trip total: 5,250.00 MAD", result["final_answer"])
        self.assertIn("Marrakech.pdf, page 7", result["final_answer"])

    @patch("app.graph.nodes.ENABLE_ANSWER_VALIDATION", True)
    @patch("app.graph.nodes._invoke_chat_model")
    def test_budget_conflict_needs_revision(self, invoke_model) -> None:
        """Reject an explanation that contradicts deterministic trip totals."""
        result = validate_answer_node(
            {
                "question": (
                    "Estimate the budget for three days in Marrakech for a "
                    "moderate traveler."
                ),
                "intent": "budget",
                "selected_path": "budget",
                "retrieved_documents": [
                    Document(
                        page_content="Moderate prices.",
                        metadata={"filename": "Marrakech.pdf", "page": 7},
                    )
                ],
                "context": "[Source 1: Marrakech.pdf, page 7] Moderate prices.",
                "budget_result": {
                    "days": 3,
                    "daily_total_mad": 1750.0,
                    "accommodation_total_mad": 2700.0,
                    "food_total_mad": 1050.0,
                    "local_transport_total_mad": 450.0,
                    "activities_total_mad": 1050.0,
                    "intercity_transport_mad": 0.0,
                    "total_budget_mad": 5250.0,
                },
                "final_answer": (
                    "Accommodation: 2,700.00 MAD\n"
                    "Food: 1,050.00 MAD\n"
                    "Local transport: 450.00 MAD\n"
                    "Activities: 1,050.00 MAD\n"
                    "Intercity transport: 0.00 MAD\n"
                    "Daily total: 1,750.00 MAD\n"
                    "Trip total: 5,000.00 MAD\n"
                    "Source: Marrakech.pdf, page 7"
                ),
            }
        )

        self.assertEqual(result["validation_result"], "needs_revision")
        self.assertIn("trip total conflicts", result["validation_feedback"])
        invoke_model.assert_not_called()

    @patch("app.graph.nodes._invoke_chat_model")
    def test_missing_budget_result_is_explicit(self, invoke_model) -> None:
        """State clearly that a missing deterministic budget cannot be used."""
        result = generate_answer_node(
            {
                "question": "Estimate a three-day Marrakech budget.",
                "intent": "budget",
                "selected_path": "budget",
                "budget_result": "",
                "revision_count": 0,
            }
        )

        self.assertIn("budget cannot be calculated", result["final_answer"])
        invoke_model.assert_not_called()

    @patch("app.graph.nodes.ENABLE_ANSWER_VALIDATION", True)
    @patch("app.graph.nodes._invoke_chat_model")
    def test_comparison_without_sources_needs_revision(
        self,
        invoke_model,
    ) -> None:
        """Reject an unsourced Marrakech versus Chefchaouen comparison."""
        result = validate_answer_node(
            {
                "question": "Compare Marrakech and Chefchaouen.",
                "intent": "comparison",
                "selected_path": "comparison",
                "retrieved_documents": [
                    Document(
                        page_content="Grounded comparison facts.",
                        metadata={"filename": "Marrakech.pdf", "page": 2},
                    )
                ],
                "context": "[Source 1: Marrakech.pdf, page 2] Facts.",
                "comparison_result": {
                    "destination_a": "Marrakech",
                    "destination_b": "Chefchaouen",
                },
                "final_answer": "Marrakech is warmer than Chefchaouen.",
            }
        )

        self.assertEqual(result["validation_result"], "needs_revision")
        self.assertIn("source", result["validation_feedback"].lower())
        invoke_model.assert_not_called()

    @patch("app.graph.nodes.ENABLE_ANSWER_VALIDATION", True)
    @patch("app.graph.nodes._invoke_chat_model")
    def test_transport_without_sources_needs_revision(
        self,
        invoke_model,
    ) -> None:
        """Reject unsupported Tangier-to-Chefchaouen transport claims."""
        result = validate_answer_node(
            {
                "question": "How can I travel from Tangier to Chefchaouen?",
                "intent": "transport",
                "selected_path": "transport",
                "retrieved_documents": [
                    Document(
                        page_content="Grounded route facts.",
                        metadata={
                            "filename": "Transportation in Morocco.pdf",
                            "page": 3,
                        },
                    )
                ],
                "context": (
                    "[Source 1: Transportation in Morocco.pdf, page 3] Facts."
                ),
                "transport_result": {
                    "origin": "Tangier",
                    "destination": "Chefchaouen",
                },
                "final_answer": "Take a direct airport train for two hours.",
            }
        )

        self.assertEqual(result["validation_result"], "needs_revision")
        self.assertIn("source", result["validation_feedback"].lower())
        invoke_model.assert_not_called()

    @patch("app.graph.nodes.ENABLE_ANSWER_VALIDATION", False)
    @patch("app.graph.nodes._invoke_gemini")
    def test_disabled_validation_accepts_without_gemini(
        self,
        invoke_gemini,
    ) -> None:
        """Accept a generated answer locally when validation is disabled."""
        result = validate_answer_node(
            {
                "question": "What should I see?",
                "intent": "factual",
                "retrieved_documents": [
                    Document(
                        page_content="A grounded fact.",
                        metadata={"filename": "guide.pdf", "page": 1},
                    )
                ],
                "context": "[guide.pdf, page 1] A grounded fact.",
                "final_answer": "A grounded answer (guide.pdf, page 1).",
            }
        )

        self.assertEqual(result["validation_result"], "valid")
        self.assertIsInstance(result["messages"][0], AIMessage)
        invoke_gemini.assert_not_called()

    def test_quota_error_is_clear_and_not_retried(self) -> None:
        """Translate a quota error while preserving its original cause."""
        quota_error = FakeQuotaError("HTTP 429 RESOURCE_EXHAUSTED")

        with patch("app.graph.nodes.get_chat_model") as get_model:
            get_model.return_value.invoke.side_effect = quota_error
            with self.assertRaisesRegex(
                RuntimeError,
                "Gemini quota was exceeded",
            ) as raised:
                _invoke_gemini([], "answer generation")

        self.assertIs(raised.exception.__cause__, quota_error)
        get_model.return_value.invoke.assert_called_once()

    @patch("app.graph.nodes.time.sleep")
    def test_temporary_error_retries_at_most_three_attempts(
        self,
        sleep,
    ) -> None:
        """Retry 503 responses twice with bounded exponential backoff."""
        temporary_error = FakeTemporaryError("503 UNAVAILABLE: high demand")
        expected_response = AIMessage(content="Recovered answer")

        with patch("app.graph.nodes.get_chat_model") as get_model:
            get_model.return_value.invoke.side_effect = [
                temporary_error,
                temporary_error,
                expected_response,
            ]
            response = _invoke_gemini([], "answer generation")

        self.assertIs(response, expected_response)
        self.assertEqual(get_model.return_value.invoke.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [2, 4],
        )

    @patch("app.graph.nodes.time.sleep")
    def test_client_errors_are_not_retried(self, sleep) -> None:
        """Never retry invalid, authentication, or missing-model errors."""
        for status_code in (400, 401, 403, 404):
            with self.subTest(status_code=status_code):
                client_error = FakeClientError(
                    f"HTTP {status_code}: model UNAVAILABLE",
                    status_code,
                )
                with patch("app.graph.nodes.get_chat_model") as get_model:
                    get_model.return_value.invoke.side_effect = client_error
                    with self.assertRaises(RuntimeError) as raised:
                        _invoke_gemini([], "answer generation")

                self.assertIs(raised.exception.__cause__, client_error)
                get_model.return_value.invoke.assert_called_once()
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
