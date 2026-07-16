"""Define the processing nodes used by the tourism-agent graph."""

import logging
import re
import time
from collections.abc import Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from app.graph.state import TourismAgentState
from app.llm.model import get_chat_model
from app.rag.retriever import retrieve_documents
from app.tools.retrieval_tool import (
    NO_DOCUMENTS_MESSAGE,
    format_tourism_documents,
)
from app.utils.config import ConfigurationError, ENABLE_ANSWER_VALIDATION


LOGGER = logging.getLogger(__name__)
VALIDATION_RESULTS = {
    "valid",
    "needs_revision",
    "insufficient_information",
}
GEMINI_QUOTA_STATUS_CODE = 429
MAX_GEMINI_ATTEMPTS = 3
GEMINI_RETRY_DELAYS_SECONDS = (2, 4, 8)
NON_RETRYABLE_GEMINI_STATUS_CODES = {400, 401, 403, 404}
INTENT_RULES = (
    (
        "itinerary",
        (
            "plan",
            "itinerary",
            "day trip",
            "days in",
            "trip plan",
        ),
    ),
    (
        "comparison",
        (
            "compare",
            "comparison",
            "versus",
            "vs",
            "difference",
            "better than",
        ),
    ),
    (
        "budget",
        (
            "budget",
            "cost",
            "price",
            "estimate",
            "afford",
            "expensive",
            "cheap",
            "how much",
        ),
    ),
    (
        "transport",
        (
            "transport",
            "transportation",
            "travel from",
            "get from",
            "train",
            "bus",
            "taxi",
            "flight",
            "transfer",
        ),
    ),
)
RECOMMENDATION_KEYWORDS = (
    "which city",
    "which destination",
    "where should i go",
    "where should i visit",
    "recommend a city",
    "recommend a destination",
    "destination recommendation",
    "best city for me",
)
FACTUAL_KEYWORDS = (
    "attraction",
    "attractions",
    "landmark",
    "landmarks",
    "fact",
    "facts",
    "history",
    "famous for",
    "what to see",
    "things to do",
    "best time",
    "where is",
    "tell me about",
)
PREFERENCE_PATTERNS = {
    "cultural": r"\b(?:cultural|culture|historic|historical)\b",
    "beach": r"\b(?:beach|beaches|coastal|seaside)\b",
    "desert": r"\b(?:desert|sahara)\b",
    "hiking": r"\b(?:hiking|hike|trekking|trek)\b",
    "quiet": r"\b(?:quiet|peaceful|calm|relaxing)\b",
    "luxury": r"\b(?:luxury|luxurious|premium|high-end)\b",
    "moderate budget": r"\b(?:moderate|mid-range|midrange)\s+budget\b",
    "budget": r"\b(?:budget|budget-friendly|affordable|low-cost)\b",
    "family": r"\b(?:family|family-friendly|children|kids)\b",
    "couple": r"\b(?:couple|romantic|honeymoon)\b",
    "solo": r"\b(?:solo|alone|independent traveler)\b",
}


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


def _contains_keyword(text: str, keyword: str) -> bool:
    """Return whether normalized text contains a complete keyword phrase."""
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text))


def classify_tourism_intent(question: str) -> str:
    """Classify a tourism question deterministically using ordered rules."""
    normalized_question = re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-z0-9\s]", " ", question.lower()),
    ).strip()

    for intent, keywords in INTENT_RULES:
        if any(
            _contains_keyword(normalized_question, keyword)
            for keyword in keywords
        ):
            return intent

    if any(
        _contains_keyword(normalized_question, keyword)
        for keyword in RECOMMENDATION_KEYWORDS
    ):
        return "general"
    if any(
        _contains_keyword(normalized_question, keyword)
        for keyword in FACTUAL_KEYWORDS
    ):
        return "factual"
    return "general"


def extract_user_preferences(
    question: str,
    existing_preferences: Sequence[str] | None = None,
) -> list[str]:
    """Merge simple stated travel preferences with prior thread values."""
    normalized_question = question.lower().strip()
    preferences = list(dict.fromkeys(existing_preferences or []))
    detected_preferences: list[str] = []
    moderate_budget_mentioned = bool(
        re.search(
            PREFERENCE_PATTERNS["moderate budget"],
            normalized_question,
        )
    )

    for preference, pattern in PREFERENCE_PATTERNS.items():
        if preference == "budget" and moderate_budget_mentioned:
            continue
        if not re.search(pattern, normalized_question):
            continue

        negative_pattern = (
            r"\b(?:do not|don't|no longer|not)\s+"
            r"(?:(?:prefer|like|want|enjoy)\s+)?"
            r"(?:(?:a|an)\s+)?"
            f"(?:{pattern})"
            r"|\bavoid(?:ing)?\s+"
            f"(?:{pattern})"
        )
        if re.search(negative_pattern, normalized_question):
            preferences = [
                value for value in preferences if value != preference
            ]
            continue

        detected_preferences.append(preference)

    for preference in detected_preferences:
        if preference not in preferences:
            preferences.append(preference)

    return preferences


def _format_conversation_history(
    messages: Sequence[BaseMessage],
    current_question: str,
) -> str:
    """Format accepted prior turns while excluding the current question."""
    previous_messages = list(messages)
    if (
        previous_messages
        and isinstance(previous_messages[-1], HumanMessage)
        and _message_text(previous_messages[-1]) == current_question
    ):
        previous_messages.pop()

    if not previous_messages:
        return "No previous conversation."

    formatted_messages: list[str] = []
    for message in previous_messages:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        formatted_messages.append(f"{role}: {_message_text(message)}")
    return "\n".join(formatted_messages)


def _invoke_gemini(
    messages: list[BaseMessage],
    operation: str,
) -> BaseMessage:
    """Invoke Gemini with bounded retries for temporary availability errors."""
    try:
        model = get_chat_model()
    except Exception as exc:
        raise RuntimeError(f"Gemini {operation} failed: {exc}") from exc

    for attempt in range(1, MAX_GEMINI_ATTEMPTS + 1):
        try:
            return model.invoke(
                messages,
                automatic_function_calling={"disable": True},
            )
        except Exception as exc:
            if _is_gemini_quota_error(exc):
                raise RuntimeError(
                    "Gemini quota was exceeded while performing "
                    f"{operation}. Wait for the quota window to reset or use "
                    "another configured model."
                ) from exc

            retryable = _is_retryable_gemini_error(exc)
            if retryable and attempt < MAX_GEMINI_ATTEMPTS:
                delay = GEMINI_RETRY_DELAYS_SECONDS[attempt - 1]
                LOGGER.warning(
                    "Gemini is temporarily unavailable during %s "
                    "(attempt %d/%d). Retrying in %d seconds.",
                    operation,
                    attempt,
                    MAX_GEMINI_ATTEMPTS,
                    delay,
                )
                time.sleep(delay)
                continue
            if retryable:
                raise RuntimeError(
                    "Gemini is temporarily unavailable while performing "
                    f"{operation} after {MAX_GEMINI_ATTEMPTS} attempts. "
                    "Please try again later."
                ) from exc
            raise RuntimeError(
                f"Gemini {operation} failed: {exc}"
            ) from exc

    raise RuntimeError(f"Gemini {operation} failed unexpectedly.")


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """Return an exception and its unique chained causes in order."""
    exceptions: list[BaseException] = []
    current: BaseException | None = exc
    visited: set[int] = set()

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        exceptions.append(current)
        current = current.__cause__ or current.__context__
    return exceptions


def _exception_status_codes(exceptions: Sequence[BaseException]) -> set[int]:
    """Collect numeric HTTP-like status codes from an exception chain."""
    status_codes: set[int] = set()
    for exception in exceptions:
        candidates = (
            getattr(exception, "code", None),
            getattr(exception, "status_code", None),
            getattr(
                getattr(exception, "response", None),
                "status_code",
                None,
            ),
        )
        for candidate in candidates:
            if isinstance(candidate, int):
                status_codes.add(candidate)
    return status_codes


def _is_retryable_gemini_error(exc: BaseException) -> bool:
    """Retry only 503, UNAVAILABLE, or high-demand service failures."""
    exceptions = _exception_chain(exc)
    status_codes = _exception_status_codes(exceptions)
    if status_codes & NON_RETRYABLE_GEMINI_STATUS_CODES:
        return False
    if any(
        isinstance(exception, (ConfigurationError, TypeError, ValueError))
        for exception in exceptions
    ):
        return False

    error_text = " ".join(str(exception) for exception in exceptions).upper()
    return (
        503 in status_codes
        or bool(re.search(r"\b503\b", error_text))
        or "UNAVAILABLE" in error_text
        or "HIGH DEMAND" in error_text
    )


def _is_gemini_quota_error(exc: BaseException) -> bool:
    """Return whether an exception chain represents a Gemini quota error."""
    exceptions = _exception_chain(exc)
    status_codes = _exception_status_codes(exceptions)
    error_text = " ".join(str(exception) for exception in exceptions).upper()
    return (
        GEMINI_QUOTA_STATUS_CODE in status_codes
        or "RESOURCE_EXHAUSTED" in error_text
        or bool(re.search(r"\bHTTP(?:/[0-9.]+)?\s+429\b", error_text))
    )


def _normalize_choice(
    output: str,
    allowed_values: set[str],
    fallback: str,
) -> str:
    """Normalize a model label while rejecting explanatory output."""
    normalized = re.sub(r"[^a-z_]", "", output.lower().strip())
    return normalized if normalized in allowed_values else fallback


def classify_intent_node(state: TourismAgentState) -> dict:
    """Classify intent without consuming a Gemini API request."""
    question = _require_question(state)
    intent = classify_tourism_intent(question)
    user_preferences = extract_user_preferences(
        question,
        state.get("user_preferences", []),
    )
    updates: dict[str, object] = {
        "intent": intent,
        "user_preferences": user_preferences,
        "validation_result": "",
        "revision_count": 0,
    }
    messages = state.get("messages", [])
    if not (
        messages
        and isinstance(messages[-1], HumanMessage)
        and _message_text(messages[-1]) == question
    ):
        updates["messages"] = [HumanMessage(content=question)]
    return updates


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
    user_preferences = state.get("user_preferences", [])
    conversation_history = _format_conversation_history(
        state.get("messages", []),
        question,
    )
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
                    "Use stored preferences when relevant, but never let them "
                    "override or invent facts beyond the retrieved context. "
                    f"{structure_rule}"
                )
            ),
            HumanMessage(
                content=(
                    f"Intent: {intent}\n"
                    f"Question: {question}\n\n"
                    "Previous conversation:\n"
                    f"{conversation_history}\n\n"
                    "Stored user preferences:\n"
                    f"{', '.join(user_preferences) or 'None'}\n\n"
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

    return {
        "final_answer": answer,
        "revision_count": revision_count,
    }


def validate_answer_node(state: TourismAgentState) -> dict:
    """Validate answer coverage, grounding, sources, and intent structure."""
    question = _require_question(state)
    documents = state.get("retrieved_documents", [])
    context = state.get("context", "").strip()
    answer = state.get("final_answer", "").strip()
    intent = state.get("intent", "general")

    if not ENABLE_ANSWER_VALIDATION:
        result: dict[str, object] = {"validation_result": "valid"}
        if answer:
            result["messages"] = [AIMessage(content=answer)]
        return result

    if not documents or not context or context == NO_DOCUMENTS_MESSAGE:
        result: dict[str, object] = {
            "validation_result": "insufficient_information"
        }
        if answer:
            result["messages"] = [AIMessage(content=answer)]
        return result
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
    result: dict[str, object] = {
        "validation_result": validation_result
    }
    if validation_result in {"valid", "insufficient_information"}:
        result["messages"] = [AIMessage(content=answer)]
    return result
