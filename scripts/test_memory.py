"""Exercise thread-scoped LangGraph conversation memory."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import sys
import time
from uuid import uuid4

from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import (
    classify_tourism_intent,
    extract_user_preferences,
)
from app.graph.state import TourismAgentState
from app.graph.workflow import build_graph
from app.utils.logger import configure_application_logging


DEFAULT_INVOCATION_DELAY_SECONDS = 15.0
PREFERENCE_QUESTION = (
    "I prefer quiet cultural destinations with a moderate budget."
)
FOLLOW_UP_QUESTION = "Which city should I visit?"
EXPECTED_PREFERENCES = {"quiet", "cultural", "moderate budget"}
TEMPORARY_GEMINI_MARKERS = (
    "429",
    "503",
    "high demand",
    "quota",
    "rate limit",
    "resource exhausted",
    "resource_exhausted",
    "temporarily unavailable",
    "unavailable",
)


@dataclass
class TestStats:
    """Track graph execution and memory assertion outcomes separately."""

    successful_graph_steps: int = 0
    failed_graph_steps: int = 0
    external_api_interruptions: int = 0
    execution_failures: int = 0
    memory_logic_failures: int = 0
    completed_memory_checks: int = 0


class DelayedGraphInvoker:
    """Invoke a graph while spacing consecutive requests predictably."""

    def __init__(
        self,
        graph: CompiledStateGraph,
        delay_seconds: float,
    ) -> None:
        self._graph = graph
        self._delay_seconds = delay_seconds
        self._invocation_count = 0

    def invoke(self, question: str, thread_id: str) -> TourismAgentState:
        """Invoke one question after waiting when a prior call was made."""
        if self._invocation_count:
            print(
                "Waiting "
                f"{self._delay_seconds:g} seconds before the next graph "
                "invocation..."
            )
            time.sleep(self._delay_seconds)

        try:
            return _invoke_question(self._graph, question, thread_id)
        finally:
            self._invocation_count += 1


def _non_negative_float(value: str) -> float:
    """Parse a non-negative floating-point command-line value."""
    parsed_value = float(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed_value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse memory-test mode and request-spacing options."""
    parser = argparse.ArgumentParser(
        description="Test LangGraph preference memory and thread isolation."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--same-thread-only",
        action="store_true",
        help="Run only the same-thread preference continuity scenario.",
    )
    mode_group.add_argument(
        "--different-thread-only",
        action="store_true",
        help="Run only the fresh-thread isolation scenario.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=_non_negative_float,
        default=DEFAULT_INVOCATION_DELAY_SECONDS,
        help=(
            "Seconds to wait between graph invocations "
            f"(default: {DEFAULT_INVOCATION_DELAY_SECONDS:g})."
        ),
    )
    return parser.parse_args(argv)


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


def _is_temporary_gemini_error(exc: BaseException) -> bool:
    """Return whether an exception chain indicates quota or availability."""
    messages: list[str] = []
    current: BaseException | None = exc
    visited: set[int] = set()

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        messages.append(str(current).casefold())
        current = current.__cause__ or current.__context__

    combined_message = " ".join(messages)
    return any(
        marker in combined_message for marker in TEMPORARY_GEMINI_MARKERS
    )


def _run_graph_step(
    invoker: DelayedGraphInvoker,
    question: str,
    thread_id: str,
    label: str,
    stats: TestStats,
) -> TourismAgentState | None:
    """Run one graph step and classify external failures separately."""
    try:
        result = invoker.invoke(question, thread_id)
    except Exception as exc:
        stats.failed_graph_steps += 1
        if _is_temporary_gemini_error(exc):
            stats.external_api_interruptions += 1
            print(
                f"{label} interrupted by temporary Gemini quota or "
                f"availability limits: {exc}"
            )
            print(
                "This is an external API interruption, not a "
                "memory-logic failure."
            )
        else:
            stats.execution_failures += 1
            print(f"{label} failed: {exc}")
        return None

    stats.successful_graph_steps += 1
    _print_result(label, result)
    return result


def _run_same_thread_check(
    invoker: DelayedGraphInvoker,
    stats: TestStats,
) -> None:
    """Verify preferences survive a follow-up on one unique thread."""
    thread_id = f"tourism-same-thread-{uuid4()}"
    setup_result = _run_graph_step(
        invoker,
        PREFERENCE_QUESTION,
        thread_id,
        "Preference setup",
        stats,
    )
    if setup_result is None:
        print("Same-thread follow-up skipped because setup did not complete.")
        return

    remembered_result = _run_graph_step(
        invoker,
        FOLLOW_UP_QUESTION,
        thread_id,
        "Same-thread follow-up",
        stats,
    )
    if remembered_result is None:
        return

    stats.completed_memory_checks += 1
    remembered_preferences = set(
        remembered_result.get("user_preferences", [])
    )
    if EXPECTED_PREFERENCES.issubset(remembered_preferences):
        print(
            "Same-thread memory check passed: cultural, quiet, and "
            "moderate budget preferences were preserved."
        )
        return

    missing_preferences = sorted(
        EXPECTED_PREFERENCES.difference(remembered_preferences)
    )
    stats.memory_logic_failures += 1
    print(
        "Memory-logic failure: same-thread preferences were missing: "
        + ", ".join(missing_preferences)
    )


def _run_different_thread_check(
    invoker: DelayedGraphInvoker,
    stats: TestStats,
) -> None:
    """Verify a unique thread cannot see preferences from another thread."""
    thread_id = f"tourism-different-thread-{uuid4()}"
    isolated_result = _run_graph_step(
        invoker,
        FOLLOW_UP_QUESTION,
        thread_id,
        "Different-thread follow-up",
        stats,
    )
    if isolated_result is None:
        return

    stats.completed_memory_checks += 1
    isolated_preferences = isolated_result.get("user_preferences", [])
    if not isolated_preferences:
        print(
            "Different-thread memory check passed: the new thread has no "
            "previous preferences."
        )
        return

    stats.memory_logic_failures += 1
    print(
        "Memory-logic failure: preferences leaked into a new thread: "
        f"{isolated_preferences}"
    )


def _print_summary(
    stats: TestStats,
    required_memory_checks: int,
    local_checks_passed: bool,
) -> int:
    """Print the result summary and return the process exit code."""
    print("\n" + "=" * 72)
    print("Memory test summary")
    print(f"Successful graph steps: {stats.successful_graph_steps}")
    print(f"Failed graph steps: {stats.failed_graph_steps}")
    print(f"External API interruptions: {stats.external_api_interruptions}")
    print(f"Memory-logic failures: {stats.memory_logic_failures}")
    print(
        "Completed memory checks: "
        f"{stats.completed_memory_checks}/{required_memory_checks}"
    )

    all_checks_completed = (
        stats.completed_memory_checks == required_memory_checks
    )
    passed = (
        local_checks_passed
        and all_checks_completed
        and stats.memory_logic_failures == 0
        and stats.execution_failures == 0
    )
    if passed:
        print("Memory test passed.")
        return 0

    if stats.external_api_interruptions and not all_checks_completed:
        print(
            "Memory test was incomplete because Gemini quota or "
            "availability interrupted a required graph call."
        )
    else:
        print("Memory test did not complete all required checks successfully.")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Verify preference continuity, thread isolation, or both."""
    args = _parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    configure_application_logging()

    local_preferences = set(extract_user_preferences(PREFERENCE_QUESTION))
    local_intent = classify_tourism_intent(FOLLOW_UP_QUESTION)
    local_checks_passed = (
        EXPECTED_PREFERENCES.issubset(local_preferences)
        and local_intent == "general"
    )
    print("Deterministic preference and intent checks:", end=" ")
    print("passed" if local_checks_passed else "failed")

    stats = TestStats()
    if not local_checks_passed:
        stats.memory_logic_failures += 1

    graph = build_graph()
    invoker = DelayedGraphInvoker(graph, args.delay_seconds)

    run_same_thread = not args.different_thread_only
    run_different_thread = not args.same_thread_only
    required_memory_checks = int(run_same_thread) + int(run_different_thread)

    if run_same_thread:
        _run_same_thread_check(invoker, stats)
    if run_different_thread:
        _run_different_thread_check(invoker, stats)

    return _print_summary(
        stats,
        required_memory_checks,
        local_checks_passed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
