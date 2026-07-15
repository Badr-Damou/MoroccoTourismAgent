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


if __name__ == "__main__":
    unittest.main()
