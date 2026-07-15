"""Create the Gemini chat model for future agent workflows."""

from langchain_google_genai import ChatGoogleGenerativeAI

from app.utils.config import GEMINI_CHAT_MODEL, get_google_api_key


def get_chat_model() -> ChatGoogleGenerativeAI:
    """Return a reusable Gemini chat client without invoking the model."""
    return ChatGoogleGenerativeAI(
        model=GEMINI_CHAT_MODEL,
        temperature=0.2,
        api_key=get_google_api_key(),
        max_retries=1,
    )
