"""Run representative questions through every intent-based graph route."""

import argparse
from collections.abc import Sequence
import json
import sys
import time
from uuid import uuid4

from app.graph.state import TourismAgentState
from app.graph.workflow import build_graph
from app.utils.logger import configure_application_logging


DEFAULT_PAUSE_SECONDS = 10.0
ROUTING_CASES = (
    (
        "What are the main tourist attractions in Marrakech?",
        "factual",
    ),
    (
        "Plan a two-day trip to Chefchaouen.",
        "itinerary",
    ),
    (
        "Compare Marrakech and Chefchaouen.",
        "comparison",
    ),
    (
        "Estimate the budget for three days in Marrakech for a moderate "
        "traveler.",
        "budget",
    ),
    (
        "How can I travel from Tangier to Chefchaouen?",
        "transport",
    ),
)
SPECIALIZED_RESULT_FIELDS = {
    "itinerary": "itinerary_result",
    "comparison": "comparison_result",
    "budget": "budget_result",
    "transport": "transport_result",
}
EXPECTED_BUDGET_RESULT = {
    "days": 3,
    "daily_total_mad": 1750.0,
    "accommodation_total_mad": 2700.0,
    "food_total_mad": 1050.0,
    "local_transport_total_mad": 450.0,
    "activities_total_mad": 1050.0,
    "intercity_transport_mad": 0.0,
    "total_budget_mad": 5250.0,
}
TEMPORARY_GEMINI_MARKERS = (
    "429",
    "503",
    "high demand",
    "quota",
    "resource_exhausted",
    "temporarily unavailable",
    "unavailable",
)


def _non_negative_float(value: str) -> float:
    """Parse a non-negative command-line number."""
    parsed_value = float(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed_value


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the delay between consecutive graph invocations."""
    parser = argparse.ArgumentParser(
        description="Test every deterministic LangGraph intent route."
    )
    parser.add_argument(
        "--pause-seconds",
        type=_non_negative_float,
        default=DEFAULT_PAUSE_SECONDS,
        help="Seconds to wait between graph calls (default: 10).",
    )
    return parser.parse_args(arguments)


def _exception_text(exc: BaseException) -> str:
    """Combine an exception chain into safe diagnostic text."""
    messages: list[str] = []
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        messages.append(str(current))
        current = current.__cause__ or current.__context__
    return " ".join(messages)


def _is_temporary_gemini_error(exc: BaseException) -> bool:
    """Identify quota or service-availability interruptions."""
    error_text = _exception_text(exc).casefold()
    return any(marker in error_text for marker in TEMPORARY_GEMINI_MARKERS)


def _specialized_result(
    result: TourismAgentState,
    selected_path: str,
) -> dict[str, object] | str:
    """Return the tool result associated with one selected route."""
    field_name = SPECIALIZED_RESULT_FIELDS.get(selected_path)
    if field_name is None:
        return "No specialized tool is required for this direct answer path."
    specialized_result = result.get(field_name)
    return specialized_result or "No specialized tool result was returned."


def _format_result(result: dict[str, object] | str) -> str:
    """Format a specialized result for readable terminal output."""
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)


def _case_specific_assertion(
    selected_path: str,
    specialized_result: dict[str, object] | str,
    answer: str,
) -> tuple[bool, str]:
    """Check deterministic values for the requested regression routes."""
    if not isinstance(specialized_result, dict):
        return False, "The specialized result is not structured."
    if selected_path == "comparison":
        passed = (
            specialized_result.get("destination_a") == "Marrakech"
            and specialized_result.get("destination_b") == "Chefchaouen"
        )
        return passed, "Comparison endpoints must be Marrakech and Chefchaouen."
    if selected_path == "transport":
        passed = (
            specialized_result.get("origin") == "Tangier"
            and specialized_result.get("destination") == "Chefchaouen"
        )
        return passed, "Transport endpoints must be Tangier and Chefchaouen."
    if selected_path == "budget":
        exact_result = specialized_result == EXPECTED_BUDGET_RESULT
        exact_answer_values = all(
            f"{float(value):,.2f} MAD" in answer
            for field, value in EXPECTED_BUDGET_RESULT.items()
            if field != "days"
        )
        return (
            exact_result and exact_answer_values,
            "Budget tool fields and rendered totals must match exactly.",
        )
    return True, ""


def main(arguments: Sequence[str] | None = None) -> int:
    """Run all routing cases while preserving results after failures."""
    args = _parse_args(arguments)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    configure_application_logging()

    graph = build_graph()
    successful_tests = 0
    failed_tests = 0

    for test_number, (question, expected_path) in enumerate(
        ROUTING_CASES,
        start=1,
    ):
        if test_number > 1 and args.pause_seconds:
            print(
                f"\nWaiting {args.pause_seconds:g} seconds before the next "
                "routing test..."
            )
            time.sleep(args.pause_seconds)

        print("\n" + "=" * 72)
        print(f"Question: {question}")
        try:
            result = graph.invoke(
                {
                    "question": question,
                    "revision_count": 0,
                },
                config={
                    "configurable": {
                        "thread_id": f"routing-test-{uuid4()}",
                    }
                },
            )
        except Exception as exc:
            failed_tests += 1
            if _is_temporary_gemini_error(exc):
                print(
                    "Routing test interrupted by temporary Gemini quota or "
                    f"availability limits: {exc}"
                )
            else:
                print(f"Routing test failed: {exc}")
            continue

        detected_intent = result.get("intent", "unknown")
        selected_path = result.get("selected_path", "unknown")
        specialized_result = _specialized_result(result, selected_path)
        specialized_result_ready = (
            selected_path not in SPECIALIZED_RESULT_FIELDS
            or (
                isinstance(specialized_result, dict)
                and specialized_result.get("status") != "missing_information"
            )
        )
        case_specific_matches, case_specific_message = (
            _case_specific_assertion(
                selected_path,
                specialized_result,
                result.get("final_answer", ""),
            )
            if selected_path in {"comparison", "budget", "transport"}
            else (True, "")
        )
        route_matches = (
            detected_intent == expected_path
            and selected_path == expected_path
            and specialized_result_ready
            and case_specific_matches
        )

        print(f"Detected intent: {detected_intent}")
        print(f"Selected path: {selected_path}")
        print(
            "Retrieved documents: "
            f"{len(result.get('retrieved_documents', []))}"
        )
        print(
            "Specialized tool result:\n"
            f"{_format_result(specialized_result)}"
        )
        print(
            "Validation result: "
            f"{result.get('validation_result', 'unknown')}"
        )
        print(f"Final answer:\n{result.get('final_answer', '')}")

        if route_matches:
            successful_tests += 1
            print("Routing assertion: passed")
        else:
            failed_tests += 1
            print(
                "Routing assertion: failed; expected intent/path "
                f"'{expected_path}' with a complete specialized result."
            )
            if case_specific_message and not case_specific_matches:
                print(f"Case-specific assertion: {case_specific_message}")

    print("\n" + "=" * 72)
    print("Routing test summary")
    print(f"Successful tests: {successful_tests}")
    print(f"Failed tests: {failed_tests}")
    return 1 if failed_tests else 0


if __name__ == "__main__":
    raise SystemExit(main())
