"""Smoke-test the configured chat provider with one short prompt."""

import sys

from app.llm.model import describe_ollama_error, get_chat_model
from app.utils.config import (
    GEMINI_CHAT_MODEL,
    LLM_PROVIDER,
    OLLAMA_CHAT_MODEL,
)


def _response_text(response: object) -> str:
    """Return readable text from a LangChain chat response."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def main() -> int:
    """Print the selected provider/model and one generated response."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    selected_model = (
        OLLAMA_CHAT_MODEL
        if LLM_PROVIDER == "ollama"
        else GEMINI_CHAT_MODEL
    )
    print(f"Selected provider: {LLM_PROVIDER}")
    print(f"Selected model: {selected_model}")

    try:
        response = get_chat_model().invoke(
            "Reply with one short sentence welcoming a visitor to Morocco."
        )
    except Exception as exc:
        if LLM_PROVIDER == "ollama":
            error_message = describe_ollama_error(exc)
        else:
            error_message = f"Gemini chat request failed: {exc}"
        print(f"Generated response: ERROR - {error_message}")
        return 1

    print(f"Generated response: {_response_text(response)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
