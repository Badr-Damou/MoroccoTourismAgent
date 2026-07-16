"""Create the configured chat model used by the agent workflow."""

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from app.utils.config import (
    GEMINI_CHAT_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    ConfigurationError,
    get_google_api_key,
)


def get_chat_model() -> BaseChatModel:
    """Return the selected chat client without invoking the model."""
    if LLM_PROVIDER == "ollama":
        return ChatOllama(
            model=OLLAMA_CHAT_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
        )
    if LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(
            model=GEMINI_CHAT_MODEL,
            temperature=0.2,
            api_key=get_google_api_key(),
            max_retries=1,
        )
    raise ConfigurationError(
        f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. "
        "Supported providers are: ollama, gemini."
    )


def describe_ollama_error(exc: BaseException) -> str:
    """Translate common Ollama failures into actionable diagnostics."""
    messages: list[str] = []
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        messages.append(str(current))
        current = current.__cause__ or current.__context__

    detail = " ".join(messages).strip()
    normalized_detail = detail.casefold()
    if any(
        marker in normalized_detail
        for marker in (
            "connection refused",
            "actively refused",
            "winerror 10061",
            "errno 111",
        )
    ):
        return (
            f"Connection refused by Ollama at {OLLAMA_BASE_URL}. "
            "Start the Ollama service and try again."
        )
    if (
        re.search(r"\b(?:404|not found)\b", normalized_detail)
        or "pull model" in normalized_detail
    ) and "model" in normalized_detail:
        return (
            f"Ollama model '{OLLAMA_CHAT_MODEL}' is not installed. "
            f"Install it with: ollama pull {OLLAMA_CHAT_MODEL}"
        )
    if any(
        marker in normalized_detail
        for marker in (
            "connecterror",
            "connection error",
            "failed to connect",
            "all connection attempts failed",
            "name or service not known",
            "timed out",
            "timeout",
        )
    ):
        return (
            f"Ollama service is unavailable at {OLLAMA_BASE_URL}. "
            "Verify that Ollama is running and the base URL is correct."
        )
    return f"Ollama chat request failed: {detail or type(exc).__name__}."
