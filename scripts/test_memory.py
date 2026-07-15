"""Exercise thread-scoped LangGraph conversation memory."""

import sys

from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import (
    classify_tourism_intent,
    extract_user_preferences,
)
from app.graph.workflow import build_graph
from app.graph.state import TourismAgentState
from app.utils.logger import configure_application_logging


PREFERENCE_QUESTION = (
    "I prefer quiet cultural destinations with a moderate budget."
)
FOLLOW_UP_QUESTION = "Which city should I visit?"


def _invoke_question(
    graph: CompiledStateGraph,
    question: str,
    thread_id: str,
) -> TourismAgentState:
    """Invoke one question using the supplied conversation thread."""
    return graph.invoke(
        {
            "question": question,
            "revision_count": 0,
        },
        config={"configurable": {"thread_id": thread_id}},
    )


def _print_result(label: str, result: TourismAgentState) -> None:
    """Print the memory values relevant to this smoke test."""
    print("\n" + "=" * 72)
    print(label)
    print(f"Stored preferences: {result.get('user_preferences', [])}")
    print(f"Detected intent: {result.get('intent', 'unknown')}")
    print(f"Message count: {len(result.get('messages', []))}")
    print(f"Final answer:\n{result.get('final_answer', '')}")


def main() -> int:
    """Verify preference continuity and isolation across two thread IDs."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    configure_application_logging()
    graph = build_graph()
    expected_preferences = {"quiet", "cultural", "moderate budget"}
    local_preferences = set(extract_user_preferences(PREFERENCE_QUESTION))
    local_intent = classify_tourism_intent(FOLLOW_UP_QUESTION)
    local_checks_passed = (
        expected_preferences.issubset(local_preferences)
        and local_intent == "general"
    )
    print("Deterministic preference and intent checks:", end=" ")
    print("passed" if local_checks_passed else "failed")

    successful_steps = 0
    failed_steps = 0
    setup_result: TourismAgentState | None = None
    remembered_result: TourismAgentState | None = None
    isolated_result: TourismAgentState | None = None

    try:
        setup_result = _invoke_question(
            graph,
            PREFERENCE_QUESTION,
            "tourism-user-1",
        )
    except Exception as exc:
        failed_steps += 1
        print(f"Preference setup failed: {exc}")
    else:
        successful_steps += 1
        _print_result("Preference setup", setup_result)

    if setup_result is not None:
        try:
            remembered_result = _invoke_question(
                graph,
                FOLLOW_UP_QUESTION,
                "tourism-user-1",
            )
        except Exception as exc:
            failed_steps += 1
            print(f"Same-thread follow-up failed: {exc}")
        else:
            successful_steps += 1
            _print_result("Same thread follow-up", remembered_result)
    else:
        print("Same-thread follow-up skipped because setup did not complete.")

    try:
        isolated_result = _invoke_question(
            graph,
            FOLLOW_UP_QUESTION,
            "tourism-user-2",
        )
    except Exception as exc:
        failed_steps += 1
        print(f"Different-thread follow-up failed: {exc}")
    else:
        successful_steps += 1
        _print_result("Different thread follow-up", isolated_result)

    memory_checks_passed = True
    if remembered_result is None:
        memory_checks_passed = False
    elif not expected_preferences.issubset(
        set(remembered_result.get("user_preferences", []))
    ):
        print("Memory test failed: expected preferences were not remembered.")
        memory_checks_passed = False
    if isolated_result is None:
        memory_checks_passed = False
    elif isolated_result.get("user_preferences", []):
        print("Memory test failed: preferences leaked between thread IDs.")
        memory_checks_passed = False

    print("\n" + "=" * 72)
    print("Memory test summary")
    print(f"Successful graph steps: {successful_steps}")
    print(f"Failed graph steps: {failed_steps}")
    if local_checks_passed and memory_checks_passed:
        print("Memory test passed: preferences stayed within the same thread.")
        return 0
    print("Memory test did not complete all thread-isolation checks.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
