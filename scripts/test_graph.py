"""Run two representative questions through the LangGraph workflow."""

import sys

from app.graph.nodes import classify_tourism_intent
from app.graph.workflow import build_graph
from app.utils.logger import configure_application_logging


TEST_QUESTIONS = (
    "What are the main tourist attractions in Marrakech?",
    "Plan a two-day trip to Chefchaouen.",
)


def main() -> int:
    """Invoke the graph for factual and itinerary tourism questions."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    configure_application_logging()
    graph = build_graph()
    successful_tests = 0
    failed_tests = 0

    for question_number, question in enumerate(TEST_QUESTIONS, start=1):
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
