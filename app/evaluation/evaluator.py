"""Deterministically score graph answers and retrieved documents."""

import json
import re
from collections.abc import Sequence

from langchain_core.documents import Document


SPECIALIZED_RESULT_FIELDS = {
    "itinerary": "itinerary_result",
    "comparison": "comparison_result",
    "budget": "budget_result",
    "transport": "transport_result",
}
KNOWN_DESTINATION_SOURCES = {
    "marrakech": "marrakech.pdf",
    "chefchaouen": "chefchaoun.pdf",
}
STOP_WORDS = {
    "about",
    "after",
    "also",
    "among",
    "answer",
    "between",
    "considering",
    "could",
    "does",
    "from",
    "have",
    "into",
    "main",
    "morocco",
    "recommend",
    "should",
    "their",
    "there",
    "these",
    "those",
    "travel",
    "traveler",
    "visit",
    "what",
    "when",
    "where",
    "which",
    "with",
}
SOURCE_REFERENCE_PATTERN = re.compile(
    r"[\w .’'-]+\.pdf[^\n]{0,50}\bpage\s+\d+",
    re.I,
)
UNAVAILABLE_PATTERNS = (
    "cannot be calculated",
    "do not have enough information",
    "not available in the retrieved",
    "unavailable in the retrieved",
)
FACT_UNIT_PATTERN = re.compile(
    r"\b\d[\d,.]*(?:\s*[–-]\s*\d[\d,.]*)?\s*"
    r"(?:MAD|°C|km/h|kilometers?|km|hours?|minutes?|mins?)\b",
    re.I,
)
NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")
DESTINATION_NAMES = (
    "Marrakech",
    "Chefchaouen",
    "Casablanca",
    "Rabat",
    "Tangier",
    "Tetouan",
    "Fez",
    "Meknes",
    "Oujda",
    "Agadir",
    "Essaouira",
    "Ouarzazate",
    "Merzouga",
    "Dakhla",
    "Laayoune",
    "Errachidia",
)
ATTRACTION_NAMES = (
    "Jemaa el-Fna",
    "Koutoubia Mosque",
    "Bahia Palace",
    "El Badi Palace",
    "Majorelle Garden",
    "Yves Saint Laurent Museum",
    "Saadian Tombs",
    "Marrakech Medina",
    "Plaza Uta el-Hammam",
    "Kasbah Museum",
    "Blue Medina",
    "Spanish Mosque",
    "Ras El Maa",
    "Akchour Waterfalls",
    "God's Bridge",
)


def _normalized_text(value: object) -> str:
    """Normalize spacing and punctuation for conservative text matching."""
    return re.sub(r"[^a-z0-9°]+", " ", str(value).casefold()).strip()


def _question_keywords(question: str) -> set[str]:
    """Extract stable content words used by deterministic relevance checks."""
    return {
        token
        for token in re.findall(r"[a-zA-Z]{4,}", question.casefold())
        if token not in STOP_WORDS
    }


def retrieved_sources(documents: Sequence[Document]) -> list[str]:
    """Return unique source labels in retrieval order."""
    sources: list[str] = []
    for document in documents:
        filename = document.metadata.get("filename", "unknown")
        page = document.metadata.get("page", "unknown")
        source = f"{filename}, page {page}"
        if source not in sources:
            sources.append(source)
    return sources


def _has_source_reference(answer: str) -> bool:
    """Return whether the answer contains a filename and page citation."""
    return bool(SOURCE_REFERENCE_PATTERN.search(answer))


def _specialized_result(
    selected_path: str,
    state: dict[str, object],
) -> dict[str, object] | None:
    """Return a complete structured result for a specialized graph path."""
    field = SPECIALIZED_RESULT_FIELDS.get(selected_path)
    if field is None:
        return None
    result = state.get(field)
    if not isinstance(result, dict):
        return None
    if result.get("status") == "missing_information":
        return None
    return result


def _answer_addresses_intent(
    question: str,
    detected_intent: str,
    selected_path: str,
    answer: str,
    state: dict[str, object],
) -> bool:
    """Check intent-specific answer structure without semantic model calls."""
    normalized_answer = answer.casefold()
    result = _specialized_result(selected_path, state)
    if selected_path == "itinerary":
        return "day 1" in normalized_answer
    if selected_path == "comparison":
        if result is None:
            return False
        destinations = (
            str(result.get("destination_a", "")).casefold(),
            str(result.get("destination_b", "")).casefold(),
        )
        return all(value and value in normalized_answer for value in destinations)
    if selected_path == "budget":
        return all(
            label in normalized_answer
            for label in (
                "accommodation",
                "food",
                "local transport",
                "activities",
                "daily total",
                "trip total",
            )
        ) or "budget cannot be calculated" in normalized_answer
    if selected_path == "transport":
        if result is None:
            return False
        endpoints = (
            str(result.get("origin", "")).casefold(),
            str(result.get("destination", "")).casefold(),
        )
        return all(value and value in normalized_answer for value in endpoints)

    keywords = _question_keywords(question)
    return bool(answer.strip()) and (
        detected_intent == "general"
        or not keywords
        or any(keyword in normalized_answer for keyword in keywords)
    )


def _tool_output_contradiction(
    selected_path: str,
    answer: str,
    state: dict[str, object],
) -> bool:
    """Detect direct contradictions with deterministic specialized output."""
    result = _specialized_result(selected_path, state)
    if selected_path not in SPECIALIZED_RESULT_FIELDS:
        return False
    if result is None:
        return True
    normalized_answer = answer.casefold()

    if selected_path == "budget":
        labels = {
            "accommodation": "accommodation_total_mad",
            "food": "food_total_mad",
            "local transport": "local_transport_total_mad",
            "activities": "activities_total_mad",
            "intercity transport": "intercity_transport_mad",
            "daily total": "daily_total_mad",
            "trip total": "total_budget_mad",
        }
        for label, field in labels.items():
            expected = result.get(field)
            if not isinstance(expected, (int, float)):
                return True
            matches = re.findall(
                rf"{re.escape(label)}\s*:\s*([\d,]+(?:\.\d+)?)\s*MAD",
                answer,
                re.I,
            )
            if not matches:
                return True
            if any(float(value.replace(",", "")) != float(expected) for value in matches):
                return True
        return False

    if selected_path == "itinerary":
        days = result.get("days")
        if not isinstance(days, int):
            return True
        return any(
            f"day {day}" not in normalized_answer
            for day in range(1, days + 1)
        ) or f"day {days + 1}" in normalized_answer

    endpoint_fields = (
        ("destination_a", "destination_b")
        if selected_path == "comparison"
        else ("origin", "destination")
    )
    return any(
        str(result.get(field, "")).casefold() not in normalized_answer
        for field in endpoint_fields
    )


def answer_quality_score(
    question: str,
    detected_intent: str,
    selected_path: str,
    answer: str,
    state: dict[str, object],
) -> tuple[int, list[str]]:
    """Score answer quality from one to five using fixed binary criteria."""
    non_empty = bool(answer.strip())
    addresses_intent = non_empty and _answer_addresses_intent(
        question,
        detected_intent,
        selected_path,
        answer,
        state,
    )
    has_sources = non_empty and _has_source_reference(answer)
    specialized_ready = (
        selected_path not in SPECIALIZED_RESULT_FIELDS
        or _specialized_result(selected_path, state) is not None
    )
    no_tool_contradiction = non_empty and not _tool_output_contradiction(
        selected_path,
        answer,
        state,
    )
    checks = (
        non_empty,
        addresses_intent,
        has_sources,
        specialized_ready,
        no_tool_contradiction,
    )
    score = max(1, sum(checks))
    comments: list[str] = []
    labels = (
        "answer is empty",
        "answer does not match the detected intent structure",
        "answer has no filename/page source reference",
        "specialized result is missing",
        "answer conflicts with deterministic tool output",
    )
    comments.extend(label for passed, label in zip(checks, labels) if not passed)
    return score, comments


def document_relevance_score(
    question: str,
    documents: Sequence[Document],
) -> tuple[int, list[str]]:
    """Score retrieved-document relevance from one to five."""
    if not documents:
        return 1, ["no documents were retrieved"]

    filenames = {
        str(document.metadata.get("filename", "")).casefold()
        for document in documents
    }
    normalized_question = question.casefold()
    mentioned_destinations = [
        destination
        for destination in KNOWN_DESTINATION_SOURCES
        if destination in normalized_question
    ]
    destination_match = bool(mentioned_destinations) and all(
        KNOWN_DESTINATION_SOURCES[destination] in filenames
        for destination in mentioned_destinations
    )
    if not mentioned_destinations:
        transportation_question = any(
            term in normalized_question
            for term in ("train", "taxi", "transport", "al boraq")
        )
        destination_match = (
            "transportation in morocco.pdf" in filenames
            if transportation_question
            else True
        )

    keywords = _question_keywords(question)
    retrieved_text = _normalized_text(
        " ".join(document.page_content for document in documents)
    )
    matched_keywords = {
        keyword for keyword in keywords if keyword in retrieved_text
    }
    coverage = len(matched_keywords) / len(keywords) if keywords else 1.0

    score = 2
    if destination_match:
        score += 1
    if coverage >= 0.25:
        score += 1
    if coverage >= 0.60:
        score += 1
    comments: list[str] = []
    if mentioned_destinations and not destination_match:
        comments.append("retrieved filenames do not cover every named destination")
    if coverage < 0.25:
        comments.append("retrieved text has low important-keyword coverage")
    return min(5, score), comments


def _unsupported_named_facts(answer: str, support: str) -> bool:
    """Conservatively detect known place facts absent from supplied support."""
    normalized_answer = _normalized_text(answer)
    normalized_support = _normalized_text(support)
    for name in (*DESTINATION_NAMES, *ATTRACTION_NAMES):
        normalized_name = _normalized_text(name)
        if normalized_name in normalized_answer and normalized_name not in normalized_support:
            return True
    return False


def hallucination_detected(
    question: str,
    answer: str,
    documents: Sequence[Document],
    state: dict[str, object],
) -> tuple[bool, list[str]]:
    """Conservatively flag unsupported factual content without using an LLM."""
    if not answer.strip() or any(
        marker in answer.casefold() for marker in UNAVAILABLE_PATTERNS
    ):
        return False, []

    comments: list[str] = []
    if not _has_source_reference(answer):
        comments.append("factual claims have no filename/page source reference")

    support = " ".join(
        [
            question,
            *(document.page_content for document in documents),
            json.dumps(
                {
                    field: state.get(field)
                    for field in SPECIALIZED_RESULT_FIELDS.values()
                    if state.get(field)
                },
                ensure_ascii=False,
            ),
        ]
    )
    support_numbers = {
        float(value.replace(",", ""))
        for value in NUMBER_PATTERN.findall(support)
    }
    unsupported_units = [
        fact
        for fact in FACT_UNIT_PATTERN.findall(answer)
        if any(
            float(value.replace(",", "")) not in support_numbers
            for value in NUMBER_PATTERN.findall(fact)
        )
    ]
    if unsupported_units:
        comments.append(
            "unsupported numeric facts: " + ", ".join(unsupported_units)
        )
    if _unsupported_named_facts(answer, support):
        comments.append("answer names a destination or attraction absent from support")
    return bool(comments), comments


def evaluate_graph_result(
    question: str,
    state: dict[str, object],
) -> dict[str, object]:
    """Return all deterministic metrics and comments for one graph result."""
    documents = state.get("retrieved_documents", [])
    if not isinstance(documents, list):
        documents = []
    answer = str(state.get("final_answer", "")).strip()
    detected_intent = str(state.get("intent", "unknown"))
    selected_path = str(state.get("selected_path", detected_intent))
    quality, quality_comments = answer_quality_score(
        question,
        detected_intent,
        selected_path,
        answer,
        state,
    )
    relevance, relevance_comments = document_relevance_score(
        question,
        documents,
    )
    hallucination, hallucination_comments = hallucination_detected(
        question,
        answer,
        documents,
        state,
    )
    comments = quality_comments + relevance_comments + hallucination_comments
    return {
        "detected_intent": detected_intent,
        "selected_path": selected_path,
        "final_answer": answer,
        "retrieved_document_count": len(documents),
        "retrieved_sources": retrieved_sources(documents),
        "validation_result": str(state.get("validation_result", "unknown")),
        "answer_quality_score": quality,
        "document_relevance_score": relevance,
        "hallucination_detected": hallucination,
        "comments": "; ".join(dict.fromkeys(comments)),
    }
