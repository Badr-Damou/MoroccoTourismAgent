"""Test deterministic graph decisions without external API requests."""

import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.graph.nodes import (
    _invoke_gemini,
    classify_intent_node,
    classify_tourism_intent,
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
            "Estimate the budget and cost for three days.": "budget",
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
