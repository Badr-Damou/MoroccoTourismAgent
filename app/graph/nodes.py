"""Define the processing nodes used by the tourism-agent graph."""

import re
from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.graph.state import TourismAgentState
from app.llm.model import get_chat_model
from app.rag.retriever import retrieve_documents
from app.tools.retrieval_tool import (
    NO_DOCUMENTS_MESSAGE,
    format_tourism_documents,
)


VALID_INTENTS = {
    "factual",
    "itinerary",
    "comparison",
    "budget",
    "transport",
    "general",
}
VALIDATION_RESULTS = {
    "valid",
    "needs_revision",
    "insufficient_information",
}
TEMPORARY_GEMINI_STATUS_CODES = {500, 502, 503, 504}


def _require_question(state: TourismAgentState) -> str:
    """Return a normalized question or raise a useful input error."""
    question = state.get("question", "").strip()
    if not question:
        raise ValueError("The tourism question cannot be empty.")
    return question


def _message_text(message: BaseMessage) -> str:
    """Convert text or structured model content into plain text."""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Sequence):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        return "\n".join(text_parts).strip()
    return str(content).strip()


def _invoke_gemini(
    messages: list[BaseMessage],
    operation: str,
) -> BaseMessage:
    """Invoke Gemini once and translate temporary service failures clearly."""
    try:
        return get_chat_model().invoke(
            messages,
            automatic_function_calling={"disable": True},
        )
    except Exception as exc:
        if _is_temporary_gemini_error(exc):
            raise RuntimeError(
                "Gemini is temporarily unavailable while performing "
                f"{operation}. Please try again later."
            ) from exc
        raise RuntimeError(f"Gemini {operation} failed: {exc}") from exc


def _is_temporary_gemini_error(exc: BaseException) -> bool:
    """Return whether an exception chain represents a temporary Gemini 5xx."""
    current: BaseException | None = exc
    visited: set[int] = set()

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        status_candidates = [
            getattr(current, "code", None),
            getattr(current, "status_code", None),
            getattr(getattr(current, "response", None), "status_code", None),
        ]
        if any(
            status in TEMPORARY_GEMINI_STATUS_CODES
            for status in status_candidates
        ):
            return True

        error_text = str(current).upper()
        status_pattern = "|".join(
            str(status) for status in TEMPORARY_GEMINI_STATUS_CODES
        )
        if re.search(
            rf"\bHTTP(?:/[0-9.]+)?\s+(?:{status_pattern})\b",
            error_text,
        ) or re.search(
            rf"\b(?:{status_pattern})\s+"
            r"(?:INTERNAL_SERVER_ERROR|BAD_GATEWAY|SERVICE_UNAVAILABLE|"
            r"GATEWAY_TIMEOUT|UNAVAILABLE)\b",
            error_text,
        ):
            return True

        current = current.__cause__ or current.__context__

    return False


def _normalize_choice(
    output: str,
    allowed_values: set[str],
    fallback: str,
) -> str:
    """Normalize a model label while rejecting explanatory output."""
    normalized = re.sub(r"[^a-z_]", "", output.lower().strip())
    return normalized if normalized in allowed_values else fallback


def classify_intent_node(state: TourismAgentState) -> dict:
    """Classify the question into one supported tourism intent."""
    question = _require_question(state)
    response = _invoke_gemini(
        [
            SystemMessage(
                content=(
                    "Classify the tourism question into exactly one label: "
                    "factual, itinerary, comparison, budget, transport, or "
                    "general. Return only the label and no explanation."
                )
            ),
            HumanMessage(content=question),
        ],
        operation="intent classification",
    )
    intent = _normalize_choice(
        _message_text(response),
        VALID_INTENTS,
        fallback="general",
    )
    return {"intent": intent}


def retrieve_node(state: TourismAgentState) -> dict:
    """Retrieve four relevant chunks and format them as grounded context."""
    question = _require_question(state)
    try:
        documents = retrieve_documents(question, number_of_results=4)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to retrieve tourism context: {exc}"
        ) from exc

    return {
        "retrieved_documents": documents,
        "context": format_tourism_documents(documents),
    }


def generate_answer_node(state: TourismAgentState) -> dict:
    """Generate or revise an answer using only the retrieved context."""
    question = _require_question(state)
    documents = state.get("retrieved_documents", [])
    context = state.get("context", "").strip()
    intent = state.get("intent", "general")
    current_revision_count = state.get("revision_count", 0)
    is_revision = state.get("validation_result") == "needs_revision"
    revision_count = current_revision_count + 1 if is_revision else 0

    if not documents or not context or context == NO_DOCUMENTS_MESSAGE:
        return {
            "final_answer": (
                "I do not have enough information in the tourism documents "
                "to answer this question."
            ),
            "revision_count": revision_count,
        }

    structure_rules = {
        "itinerary": "Organize the answer by day with clear day headings.",
        "comparison": "Organize the answer by explicit comparison criteria.",
        "budget": (
            "Separate accommodation, food, transport, and activities into "
            "clearly labeled sections."
        ),
    }
    structure_rule = structure_rules.get(
        intent,
        "Use a clear structure appropriate for the question.",
    )
    revision_instruction = ""
    if is_revision:
        revision_instruction = (
            "This is the only allowed revision. Improve grounding, coverage, "
            "source references, and intent-specific structure. The previous "
            f"answer was:\n{state.get('final_answer', '')}\n\n"
        )

    response = _invoke_gemini(
        [
            SystemMessage(
                content=(
                    "You are a Morocco tourism assistant. Answer only from "
                    "the supplied retrieved context. Do not add unsupported "
                    "facts. If the context does not contain the answer, say "
                    "that the information is unavailable. Cite sources using "
                    "the filename and page shown in the context. "
                    f"{structure_rule}"
                )
            ),
            HumanMessage(
                content=(
                    f"Intent: {intent}\n"
                    f"Question: {question}\n\n"
                    f"{revision_instruction}"
                    f"Retrieved context:\n{context}"
                )
            ),
        ],
        operation="answer generation",
    )
    answer = _message_text(response)
    if not answer:
        raise RuntimeError("Gemini answer generation returned an empty response.")

    messages = list(state.get("messages", []))
    messages.append(response)
    return {
        "final_answer": answer,
        "messages": messages,
        "revision_count": revision_count,
    }


def validate_answer_node(state: TourismAgentState) -> dict:
    """Validate answer coverage, grounding, sources, and intent structure."""
    question = _require_question(state)
    documents = state.get("retrieved_documents", [])
    context = state.get("context", "").strip()
    answer = state.get("final_answer", "").strip()
    intent = state.get("intent", "general")

    if not documents or not context or context == NO_DOCUMENTS_MESSAGE:
        return {"validation_result": "insufficient_information"}
    if not answer:
        return {"validation_result": "needs_revision"}

    response = _invoke_gemini(
        [
            SystemMessage(
                content=(
                    "Validate the proposed tourism answer. Check that it "
                    "answers the question, contains only claims grounded in "
                    "the retrieved context, mentions source filenames and "
                    "pages, and matches the requested intent structure. "
                    "Itinerary answers must be organized by day; comparison "
                    "answers by criteria; budget answers must separate "
                    "accommodation, food, transport, and activities. Return "
                    "exactly one label: valid, needs_revision, or "
                    "insufficient_information. Use insufficient_information "
                    "only when the context cannot support an answer."
                )
            ),
            HumanMessage(
                content=(
                    f"Question: {question}\n"
                    f"Intent: {intent}\n\n"
                    f"Retrieved context:\n{context}\n\n"
                    f"Proposed answer:\n{answer}"
                )
            ),
        ],
        operation="answer validation",
    )
    validation_result = _normalize_choice(
        _message_text(response),
        VALIDATION_RESULTS,
        fallback="needs_revision",
    )
    return {"validation_result": validation_result}
