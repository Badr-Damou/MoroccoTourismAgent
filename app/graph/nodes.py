"""Define the processing nodes used by the tourism-agent graph."""

import json
import logging
import re
import time
from collections.abc import Sequence
from difflib import SequenceMatcher

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
from app.tools import (
    build_itinerary,
    compare_destinations,
    estimate_trip_budget,
    prepare_transport_recommendation,
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
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
}
DESTINATION_STOP_WORDS = {
    "assistant",
    "budget",
    "can",
    "compare",
    "day",
    "days",
    "destination",
    "estimate",
    "give",
    "how",
    "i",
    "plan",
    "please",
    "recommend",
    "source",
    "tell",
    "travel",
    "trip",
    "user",
    "what",
    "which",
}
DEFAULT_COMPARISON_CRITERIA = [
    "attractions",
    "climate",
    "budget",
    "transport",
]
SPECIALIZED_RESULT_FIELDS = {
    "itinerary": "itinerary_result",
    "comparison": "comparison_result",
    "budget": "budget_result",
    "transport": "transport_result",
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


def _extract_days(text: str) -> int | None:
    """Extract a numeric or simple word-based trip duration."""
    numeric_match = re.search(r"\b(\d+)\s*(?:-|\s)?days?\b", text, re.I)
    if numeric_match:
        return int(numeric_match.group(1))

    word_pattern = "|".join(NUMBER_WORDS)
    word_match = re.search(
        rf"\b({word_pattern})\s*(?:-|\s)days?\b",
        text,
        re.I,
    )
    if word_match:
        return NUMBER_WORDS[word_match.group(1).lower()]
    return None


def _clean_destination_candidate(value: str) -> str:
    """Normalize a conservatively extracted destination candidate."""
    candidate = re.split(
        r"\b(?:based|by|during|for|using|with)\b",
        value,
        maxsplit=1,
        flags=re.I,
    )[0]
    candidate = re.sub(
        r"^(?:the\s+)?(?:city|destination|town)\s+of\s+",
        "",
        candidate,
        flags=re.I,
    )
    candidate = candidate.strip(" \t\r\n.,!?;:")
    words = candidate.split()
    while words and words[0].casefold() in DESTINATION_STOP_WORDS:
        words.pop(0)
    candidate = " ".join(words)
    if not words or len(words) > 4 or any(char.isdigit() for char in candidate):
        return ""
    if all(word.casefold() in DESTINATION_STOP_WORDS for word in words):
        return ""
    return candidate


def _append_unique_destination(
    destinations: list[str],
    candidate: str,
) -> None:
    """Append one non-empty destination without case-insensitive duplicates."""
    normalized_candidate = _clean_destination_candidate(candidate)
    if not normalized_candidate:
        return
    if normalized_candidate.casefold() in {
        destination.casefold() for destination in destinations
    }:
        return
    destinations.append(normalized_candidate)


def _extract_destinations(text: str) -> list[str]:
    """Extract destination names from common tourism-question phrases."""
    destinations: list[str] = []
    comparison_match = re.search(
        r"\bcompare\s+(.+?)\s+(?:and|versus|vs\.?)\s+(.+?)(?=[?.!,]|$)",
        text,
        re.I,
    )
    if comparison_match:
        _append_unique_destination(destinations, comparison_match.group(1))
        _append_unique_destination(destinations, comparison_match.group(2))

    route_match = re.search(
        r"\bfrom\s+(.+?)\s+to\s+(.+?)(?=[?.!,]|$)",
        text,
        re.I,
    )
    if route_match:
        _append_unique_destination(destinations, route_match.group(1))
        _append_unique_destination(destinations, route_match.group(2))

    for match in re.finditer(
        r"\b(?:in|to|visit)\s+(.+?)(?=[?.!,]|$)",
        text,
        re.I,
    ):
        _append_unique_destination(destinations, match.group(1))

    for proper_name in re.findall(
        r"\b[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’-]*"
        r"(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’-]*){0,3}\b",
        text,
    ):
        _append_unique_destination(destinations, proper_name)

    return destinations


def _previous_user_texts(
    state: TourismAgentState,
    current_question: str,
) -> list[str]:
    """Return prior user messages from newest to oldest."""
    previous_texts: list[str] = []
    for message in reversed(state.get("messages", [])):
        if not isinstance(message, HumanMessage):
            continue
        text = _message_text(message)
        if text and text != current_question:
            previous_texts.append(text)
    return previous_texts


def _extract_destinations_with_history(
    state: TourismAgentState,
    question: str,
) -> list[str]:
    """Extract unique destinations from the question and prior user turns."""
    destinations: list[str] = []
    for text in [question, *_previous_user_texts(state, question)]:
        for candidate in _extract_destinations(text):
            _append_unique_destination(destinations, candidate)
    return destinations


def _budget_level(
    question: str,
    user_preferences: Sequence[str],
) -> str:
    """Infer a broad budget level from the question and stored preferences."""
    combined_text = " ".join([question, *user_preferences]).casefold()
    if re.search(r"\b(?:luxury|luxurious|premium|high-end)\b", combined_text):
        return "luxury"
    if re.search(r"\b(?:moderate|mid-range|midrange)\b", combined_text):
        return "moderate"
    if re.search(
        r"\b(?:budget-friendly|budget|affordable|low-cost|cheap)\b",
        combined_text,
    ):
        return "budget"
    return "moderate"


def _itinerary_interests(
    question: str,
    user_preferences: Sequence[str],
) -> str:
    """Combine stored and explicitly stated non-budget interests."""
    interests = [
        preference
        for preference in user_preferences
        if preference not in {"budget", "luxury", "moderate budget"}
    ]
    interest_match = re.search(
        r"\b(?:focus(?:ed)? on|interested in|interests? include)\s+"
        r"(.+?)(?=[?.!]|$)",
        question,
        re.I,
    )
    if interest_match:
        for value in re.split(r",|\band\b", interest_match.group(1), flags=re.I):
            normalized_value = value.strip()
            if normalized_value and normalized_value not in interests:
                interests.append(normalized_value)
    return ", ".join(interests)


def _comparison_criteria(user_preferences: Sequence[str]) -> str:
    """Combine stable comparison criteria with relevant preferences."""
    criteria = DEFAULT_COMPARISON_CRITERIA.copy()
    for preference in user_preferences:
        if preference not in criteria:
            criteria.append(preference)
    return ", ".join(criteria)


def _transport_preference(question: str) -> str:
    """Extract a transport priority or use the balanced default."""
    normalized_question = question.casefold()
    preference_patterns = (
        ("fastest", r"\b(?:fastest|quickest|fast|quick)\b"),
        ("cheapest", r"\b(?:cheapest|cheap|affordable|low-cost)\b"),
        ("comfortable", r"\b(?:comfortable|comfort)\b"),
        ("balanced", r"\bbalanced\b"),
    )
    for preference, pattern in preference_patterns:
        if re.search(pattern, normalized_question):
            return preference
    return "balanced"


def _extract_budget_prices(
    context: str,
    budget_level: str,
) -> dict[str, float]:
    """Extract exact daily category prices for one stated budget level."""
    normalized_context = re.sub(r"\s+", " ", context)
    level_patterns = {
        "budget": r"Budget",
        "moderate": r"(?:Mid[- ]range|Moderate)",
        "luxury": r"Luxury",
    }
    level_pattern = level_patterns[budget_level]
    section_match = re.search(
        rf"\b{level_pattern}\s+Traveler\b(.*?)(?="
        r"\b(?:Budget|Mid[- ]range|Moderate|Luxury)\s+Traveler\b"
        r"|\bEstimated\s+Total\b|$)",
        normalized_context,
        re.I,
    )
    if not section_match:
        return {}

    section = section_match.group(1)
    category_patterns = {
        "accommodation_per_day": r"\bAccommodation\s*:\s*([\d,]+(?:\.\d+)?)\s*MAD\b",
        "food_per_day": r"\bFood\s*:\s*([\d,]+(?:\.\d+)?)\s*MAD\b",
        "local_transport_per_day": (
            r"\b(?:Local\s+)?Transport(?:ation)?\s*:\s*"
            r"([\d,]+(?:\.\d+)?)\s*MAD\b"
        ),
        "activities_per_day": r"\bActivities\s*:\s*([\d,]+(?:\.\d+)?)\s*MAD\b",
    }
    prices: dict[str, float] = {}
    for field_name, pattern in category_patterns.items():
        price_match = re.search(pattern, section, re.I)
        if price_match:
            prices[field_name] = float(price_match.group(1).replace(",", ""))
    return prices


def _normalize_location_name(value: str) -> str:
    """Normalize a destination or source filename for safe comparison."""
    without_extension = re.sub(r"\.pdf$", "", value, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", without_extension.casefold())


def _context_for_destination(context: str, destination: str) -> str:
    """Keep retrieved source chunks associated with one destination."""
    normalized_destination = _normalize_location_name(destination)
    if not normalized_destination:
        return ""

    matching_chunks: list[str] = []
    chunks = re.split(r"(?=\[Source\s+\d+:)", context)
    for chunk in chunks:
        source_match = re.match(
            r"\[Source\s+\d+:\s*([^,\]]+)",
            chunk.strip(),
            re.I,
        )
        if not source_match:
            continue
        normalized_source = _normalize_location_name(source_match.group(1))
        similarity = SequenceMatcher(
            None,
            normalized_destination,
            normalized_source,
        ).ratio()
        if (
            normalized_destination in normalized_source
            or normalized_source in normalized_destination
            or similarity >= 0.8
        ):
            matching_chunks.append(chunk.strip())
    return "\n\n".join(matching_chunks)


def _missing_information_result(
    tool_name: str,
    missing_fields: Sequence[str],
    message: str,
) -> dict[str, object]:
    """Build a consistent structured result for unavailable tool inputs."""
    return {
        "status": "missing_information",
        "tool": tool_name,
        "missing_fields": list(missing_fields),
        "message": message,
    }


def _specialized_result(state: TourismAgentState) -> dict[str, object] | str:
    """Return the result associated with the selected graph path."""
    selected_path = state.get("selected_path", state.get("intent", "general"))
    result_field = SPECIALIZED_RESULT_FIELDS.get(selected_path)
    if result_field is None:
        return "No specialized tool was required for this path."
    result = state.get(result_field)
    return result if result else "No specialized tool result is available."


def _format_specialized_result(state: TourismAgentState) -> str:
    """Format the selected tool result for Gemini prompts."""
    result = _specialized_result(state)
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)


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
        "selected_path": intent,
        "itinerary_result": "",
        "comparison_result": "",
        "budget_result": "",
        "transport_result": "",
        "validation_result": "",
        "validation_feedback": "",
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


def plan_itinerary_node(state: TourismAgentState) -> dict:
    """Prepare a deterministic itinerary request for answer generation."""
    question = _require_question(state)
    destinations = _extract_destinations_with_history(state, question)
    if not destinations:
        return {
            "selected_path": "itinerary",
            "itinerary_result": _missing_information_result(
                "build_itinerary",
                ["destination"],
                "A destination is required before an itinerary can be planned.",
            ),
        }

    days = _extract_days(question) or 1
    user_preferences = state.get("user_preferences", [])
    try:
        itinerary_result = build_itinerary.invoke(
            {
                "destination": destinations[0],
                "days": days,
                "interests": _itinerary_interests(
                    question,
                    user_preferences,
                ),
                "budget_level": _budget_level(
                    question,
                    user_preferences,
                ),
            }
        )
    except ValueError as exc:
        itinerary_result = _missing_information_result(
            "build_itinerary",
            ["destination", "days"],
            f"The itinerary inputs were invalid: {exc}",
        )

    return {
        "selected_path": "itinerary",
        "itinerary_result": itinerary_result,
    }


def compare_destinations_node(state: TourismAgentState) -> dict:
    """Prepare a deterministic destination-comparison request."""
    question = _require_question(state)
    destinations = _extract_destinations_with_history(state, question)
    if len(destinations) < 2:
        return {
            "selected_path": "comparison",
            "comparison_result": _missing_information_result(
                "compare_destinations",
                ["destination_a", "destination_b"],
                "Two different destinations are required for a comparison.",
            ),
        }

    try:
        comparison_result = compare_destinations.invoke(
            {
                "destination_a": destinations[0],
                "destination_b": destinations[1],
                "criteria": _comparison_criteria(
                    state.get("user_preferences", [])
                ),
            }
        )
    except ValueError as exc:
        comparison_result = _missing_information_result(
            "compare_destinations",
            ["destination_a", "destination_b"],
            f"The comparison inputs were invalid: {exc}",
        )

    return {
        "selected_path": "comparison",
        "comparison_result": comparison_result,
    }


def estimate_budget_node(state: TourismAgentState) -> dict:
    """Calculate a budget only from exact prices in retrieved context."""
    question = _require_question(state)
    days = _extract_days(question)
    if days is None:
        return {
            "selected_path": "budget",
            "budget_result": _missing_information_result(
                "estimate_trip_budget",
                ["days"],
                "A trip duration is required before a budget can be estimated.",
            ),
        }

    destinations = _extract_destinations(question)
    if not destinations:
        return {
            "selected_path": "budget",
            "budget_result": _missing_information_result(
                "estimate_trip_budget",
                ["destination"],
                "A destination is required to select the correct retrieved prices.",
            ),
        }

    budget_level = _budget_level(
        question,
        state.get("user_preferences", []),
    )
    destination_context = _context_for_destination(
        state.get("context", ""),
        destinations[0],
    )
    prices = _extract_budget_prices(
        destination_context,
        budget_level,
    )
    required_price_fields = (
        "accommodation_per_day",
        "food_per_day",
        "local_transport_per_day",
        "activities_per_day",
    )
    missing_fields = [
        field_name
        for field_name in required_price_fields
        if field_name not in prices
    ]
    if missing_fields:
        return {
            "selected_path": "budget",
            "budget_result": _missing_information_result(
                "estimate_trip_budget",
                missing_fields,
                f"The retrieved context does not provide every exact "
                f"{budget_level} daily price for {destinations[0]} needed "
                "for this estimate.",
            ),
        }

    try:
        budget_result = estimate_trip_budget.invoke(
            {
                "days": days,
                **prices,
            }
        )
    except ValueError as exc:
        budget_result = _missing_information_result(
            "estimate_trip_budget",
            required_price_fields,
            f"The retrieved budget values were invalid: {exc}",
        )

    return {
        "selected_path": "budget",
        "budget_result": budget_result,
    }


def transport_recommendation_node(state: TourismAgentState) -> dict:
    """Prepare a deterministic origin-to-destination transport request."""
    question = _require_question(state)
    destinations = _extract_destinations(question)
    if len(destinations) < 2:
        return {
            "selected_path": "transport",
            "transport_result": _missing_information_result(
                "prepare_transport_recommendation",
                ["origin", "destination"],
                "Both an origin and destination are required for transport advice.",
            ),
        }

    try:
        transport_result = prepare_transport_recommendation.invoke(
            {
                "origin": destinations[0],
                "destination": destinations[1],
                "preference": _transport_preference(question),
            }
        )
    except ValueError as exc:
        transport_result = _missing_information_result(
            "prepare_transport_recommendation",
            ["origin", "destination"],
            f"The transport inputs were invalid: {exc}",
        )

    return {
        "selected_path": "transport",
        "transport_result": transport_result,
    }


def generate_answer_node(state: TourismAgentState) -> dict:
    """Generate or revise an answer using only the retrieved context."""
    question = _require_question(state)
    documents = state.get("retrieved_documents", [])
    context = state.get("context", "").strip()
    intent = state.get("intent", "general")
    selected_path = state.get("selected_path", intent)
    user_preferences = state.get("user_preferences", [])
    specialized_result = _format_specialized_result(state)
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
            "Show accommodation, food, local transport, activities, daily "
            "total, and trip total in clearly labeled sections."
        ),
        "transport": (
            "Compare available transport information by travel time, cost, "
            "and convenience or comfort."
        ),
    }
    structure_rule = structure_rules.get(
        intent,
        "Use a clear structure appropriate for the question.",
    )
    revision_instruction = ""
    if is_revision:
        validation_feedback = state.get("validation_feedback", "").strip()
        revision_instruction = (
            "This is the only allowed revision. Improve grounding, coverage, "
            "source references, and intent-specific structure. The previous "
            f"answer was:\n{state.get('final_answer', '')}\n\n"
            "Validation feedback:\n"
            f"{validation_feedback or 'No detailed feedback was provided.'}"
            "\n\n"
        )

    response = _invoke_gemini(
        [
            SystemMessage(
                content=(
                    "You are a Morocco tourism assistant. Answer only from "
                    "the supplied retrieved context and specialized tool "
                    "result. Do not invent attractions, prices, routes, "
                    "schedules, travel times, or other unsupported facts. "
                    "Use calculated totals exactly as provided by the budget "
                    "tool; do not calculate replacement totals. If the "
                    "available information cannot answer the question, say "
                    "clearly what is unavailable. Cite sources using the "
                    "filename and page shown in the context. "
                    "Use stored preferences when relevant, but never let them "
                    "override or invent facts beyond the retrieved context. "
                    f"{structure_rule}"
                )
            ),
            HumanMessage(
                content=(
                    f"Intent: {intent}\n"
                    f"Selected path: {selected_path}\n"
                    f"Question: {question}\n\n"
                    "Previous conversation:\n"
                    f"{conversation_history}\n\n"
                    "Stored user preferences:\n"
                    f"{', '.join(user_preferences) or 'None'}\n\n"
                    f"{revision_instruction}"
                    "Specialized tool result:\n"
                    f"{specialized_result}\n\n"
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
    selected_path = state.get("selected_path", intent)
    specialized_result = _format_specialized_result(state)

    if not ENABLE_ANSWER_VALIDATION:
        result: dict[str, object] = {
            "validation_result": "valid",
            "validation_feedback": "",
        }
        if answer:
            result["messages"] = [AIMessage(content=answer)]
        return result

    if not documents or not context or context == NO_DOCUMENTS_MESSAGE:
        result: dict[str, object] = {
            "validation_result": "insufficient_information",
            "validation_feedback": (
                "The retrieved documents did not contain usable context."
            ),
        }
        if answer:
            result["messages"] = [AIMessage(content=answer)]
        return result
    if not answer:
        return {
            "validation_result": "needs_revision",
            "validation_feedback": "The generated answer was empty.",
        }

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
                    "accommodation, food, transport, activities, and totals; "
                    "transport answers must compare time, cost, and "
                    "convenience or comfort. Treat deterministic tool output "
                    "as authoritative for structured inputs and arithmetic. "
                    "Return "
                    "exactly one label: valid, needs_revision, or "
                    "insufficient_information. Use insufficient_information "
                    "only when the context cannot support an answer."
                )
            ),
            HumanMessage(
                content=(
                    f"Question: {question}\n"
                    f"Intent: {intent}\n"
                    f"Selected path: {selected_path}\n\n"
                    "Specialized tool result:\n"
                    f"{specialized_result}\n\n"
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
        "validation_result": validation_result,
        "validation_feedback": (
            "Revise the answer to improve question coverage, grounding, "
            "source references, and intent-specific structure."
            if validation_result == "needs_revision"
            else ""
        ),
    }
    if validation_result in {"valid", "insufficient_information"}:
        result["messages"] = [AIMessage(content=answer)]
    return result
