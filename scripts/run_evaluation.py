"""Run the fixed deterministic evaluation suite against the existing graph."""

import argparse
import csv
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from uuid import uuid4

from app.evaluation.evaluator import evaluate_graph_result
from app.evaluation.questions import COMPLEX_QUESTIONS, SIMPLE_QUESTIONS
from app.graph.workflow import build_graph
from app.utils.config import PROJECT_ROOT
from app.utils.logger import configure_application_logging


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CSV_PATH = ARTIFACTS_DIR / "evaluation_results.csv"
JSON_PATH = ARTIFACTS_DIR / "evaluation_results.json"
SUMMARY_PATH = ARTIFACTS_DIR / "evaluation_summary.json"
RESULT_FIELDS = (
    "question",
    "question_type",
    "detected_intent",
    "selected_path",
    "final_answer",
    "response_time_seconds",
    "retrieved_document_count",
    "retrieved_sources",
    "validation_result",
    "answer_quality_score",
    "document_relevance_score",
    "hallucination_detected",
    "success",
    "error_message",
    "comments",
)


def _positive_integer(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse evaluation limit and reset options."""
    parser = argparse.ArgumentParser(
        description="Evaluate the tourism graph with 20 fixed questions."
    )
    parser.add_argument(
        "--limit",
        type=_positive_integer,
        help="Run at most N pending questions for a quick evaluation.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove saved evaluation results before running.",
    )
    return parser.parse_args(arguments)


def _question_catalog() -> list[tuple[str, str]]:
    """Return the fixed evaluation questions in stable order."""
    return [
        *((question, "simple") for question in SIMPLE_QUESTIONS),
        *((question, "complex") for question in COMPLEX_QUESTIONS),
    ]


def _reset_results() -> None:
    """Remove only the three evaluation export files when requested."""
    for path in (CSV_PATH, JSON_PATH, SUMMARY_PATH):
        if path.exists():
            path.unlink()


def _load_existing_results() -> dict[str, dict[str, object]]:
    """Load valid prior records keyed by their exact question text."""
    if not JSON_PATH.exists():
        return {}
    try:
        payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"Warning: existing evaluation JSON could not be loaded: {exc}",
            file=sys.stderr,
        )
        return {}
    if not isinstance(payload, list):
        print(
            "Warning: existing evaluation JSON is not a result list; "
            "starting without resume data.",
            file=sys.stderr,
        )
        return {}
    return {
        str(item["question"]): item
        for item in payload
        if isinstance(item, dict) and isinstance(item.get("question"), str)
    }


def _ordered_results(
    results_by_question: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Return saved records in the canonical question order."""
    return [
        results_by_question[question]
        for question, _ in _question_catalog()
        if question in results_by_question
    ]


def _summary(results: Sequence[dict[str, object]]) -> dict[str, object]:
    """Calculate aggregate evaluation metrics for completed records."""
    total = len(results)
    successful = sum(bool(result.get("success")) for result in results)

    def average(field: str) -> float:
        values = [float(result.get(field, 0)) for result in results]
        return round(mean(values), 3) if values else 0.0

    def success_rate(question_type: str) -> float:
        typed_results = [
            result
            for result in results
            if result.get("question_type") == question_type
        ]
        if not typed_results:
            return 0.0
        return round(
            sum(bool(result.get("success")) for result in typed_results)
            / len(typed_results),
            4,
        )

    return {
        "total_questions": total,
        "successful_questions": successful,
        "failed_questions": total - successful,
        "average_response_time_seconds": average("response_time_seconds"),
        "average_answer_quality_score": average("answer_quality_score"),
        "average_document_relevance_score": average(
            "document_relevance_score"
        ),
        "hallucination_count": sum(
            bool(result.get("hallucination_detected")) for result in results
        ),
        "simple_question_success_rate": success_rate("simple"),
        "complex_question_success_rate": success_rate("complex"),
    }


def _save_results(
    results_by_question: dict[str, dict[str, object]],
) -> None:
    """Persist JSON, UTF-8 CSV, and summary exports after each question."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results = _ordered_results(results_by_question)
    JSON_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with CSV_PATH.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for result in results:
            csv_result = dict(result)
            csv_result["retrieved_sources"] = json.dumps(
                result.get("retrieved_sources", []),
                ensure_ascii=False,
            )
            writer.writerow(csv_result)
    SUMMARY_PATH.write_text(
        json.dumps(_summary(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _failure_result(
    question: str,
    question_type: str,
    elapsed_seconds: float,
    exc: BaseException,
) -> dict[str, object]:
    """Return a complete result record for one failed graph invocation."""
    return {
        "question": question,
        "question_type": question_type,
        "detected_intent": "unknown",
        "selected_path": "unknown",
        "final_answer": "",
        "response_time_seconds": round(elapsed_seconds, 3),
        "retrieved_document_count": 0,
        "retrieved_sources": [],
        "validation_result": "unknown",
        "answer_quality_score": 1,
        "document_relevance_score": 1,
        "hallucination_detected": False,
        "success": False,
        "error_message": str(exc),
        "comments": "Evaluation failed before a complete result was produced.",
    }


def main(arguments: Sequence[str] | None = None) -> int:
    """Run pending evaluation questions and save progress incrementally."""
    args = _parse_arguments(arguments)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    configure_application_logging()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.reset:
        _reset_results()

    results_by_question = _load_existing_results()
    graph = build_graph()
    catalog = _question_catalog()
    attempted = 0

    for question_number, (question, question_type) in enumerate(
        catalog,
        start=1,
    ):
        existing = results_by_question.get(question)
        if existing and bool(existing.get("success")):
            continue
        if args.limit is not None and attempted >= args.limit:
            break

        attempted += 1
        print(f"[{question_number}/{len(catalog)}] Running question...")
        started_at = time.perf_counter()
        try:
            state = graph.invoke(
                {"question": question, "revision_count": 0},
                config={
                    "configurable": {
                        "thread_id": f"evaluation-{uuid4()}",
                    }
                },
            )
            elapsed_seconds = time.perf_counter() - started_at
            metrics = evaluate_graph_result(question, state)
            result = {
                "question": question,
                "question_type": question_type,
                **metrics,
                "response_time_seconds": round(elapsed_seconds, 3),
                "success": bool(metrics["final_answer"]),
                "error_message": "",
            }
        except Exception as exc:
            elapsed_seconds = time.perf_counter() - started_at
            result = _failure_result(
                question,
                question_type,
                elapsed_seconds,
                exc,
            )
            print(f"Question failed: {exc}", file=sys.stderr)

        results_by_question[question] = result
        _save_results(results_by_question)
        print(
            "Completed: "
            f"success={result['success']}, "
            f"quality={result['answer_quality_score']}/5, "
            f"relevance={result['document_relevance_score']}/5"
        )

    _save_results(results_by_question)
    print(f"Evaluation results: {JSON_PATH.resolve()}")
    print(f"Evaluation CSV: {CSV_PATH.resolve()}")
    print(f"Evaluation summary: {SUMMARY_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
