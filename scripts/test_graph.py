"""Run two representative questions through the LangGraph workflow."""

import argparse
import sys
import time
from collections.abc import Sequence

from app.graph.nodes import classify_tourism_intent
from app.graph.workflow import build_graph
from app.utils.logger import configure_application_logging


TEST_QUESTIONS = (
    "What are the main tourist attractions in Marrakech?",
    "Plan a two-day trip to Chefchaouen.",
)
DEFAULT_PAUSE_SECONDS = 10.0


def _parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse an optional question and inter-question pause duration."""
    parser = argparse.ArgumentParser(
        description="Run tourism questions through the LangGraph workflow."
    )
    parser.add_argument(
        "--question",
        help="Run only this question instead of the default test suite.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=DEFAULT_PAUSE_SECONDS,
        help="Seconds to wait between questions (default: 10).",
    )
    parsed_arguments = parser.parse_args(arguments)
    if parsed_arguments.question is not None:
        parsed_arguments.question = parsed_arguments.question.strip()
        if not parsed_arguments.question:
            parser.error("--question cannot be empty.")
    if parsed_arguments.pause_seconds < 0:
        parser.error("--pause-seconds cannot be negative.")
    return parsed_arguments


def main(arguments: Sequence[str] | None = None) -> int:
    """Invoke the graph for factual and itinerary tourism questions."""
    parsed_arguments = _parse_arguments(arguments)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    configure_application_logging()
    graph = build_graph()
    questions = (
        (parsed_arguments.question,)
        if parsed_arguments.question is not None
        else TEST_QUESTIONS
    )
    successful_tests = 0
    failed_tests = 0

    for question_number, question in enumerate(questions, start=1):
        if question_number > 1 and parsed_arguments.pause_seconds:
            print(
                "\nWaiting "
                f"{parsed_arguments.pause_seconds:g} seconds before the "
                "next question..."
            )
            time.sleep(parsed_arguments.pause_seconds)
        print("\n" + "=" * 72)
        print(f"Question: {question}")
        print(
            "Deterministic intent check: "
            f"{classify_tourism_intent(question)}"
        )
        try:
            result = graph.invoke(
                {
                    "question": question,
                    "revision_count": 0,
                },
                config={
                    "configurable": {
                        "thread_id": f"graph-test-{question_number}",
                    }
                },
            )
        except Exception as exc:
            failed_tests += 1
            print(f"Test failed for this question: {exc}")
            continue

        successful_tests += 1
        print(f"Detected intent: {result.get('intent', 'unknown')}")
        print(
            "Retrieved documents: "
            f"{len(result.get('retrieved_documents', []))}"
        )
        print(
            "Validation result: "
            f"{result.get('validation_result', 'unknown')}"
        )
        print(f"Final answer:\n{result.get('final_answer', '')}")

    print("\n" + "=" * 72)
    print("Graph test summary")
    print(f"Successful tests: {successful_tests}")
    print(f"Failed tests: {failed_tests}")

    if failed_tests > 0 and successful_tests == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
