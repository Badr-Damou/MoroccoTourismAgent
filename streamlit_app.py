"""Professional Streamlit interface for the existing tourism-agent graph."""

import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import uuid4

import streamlit as st
from langchain_core.documents import Document

from app.graph.workflow import build_graph
from app.utils.config import (
    GEMINI_CHAT_MODEL,
    LLM_PROVIDER,
    OLLAMA_CHAT_MODEL,
    PROJECT_ROOT,
)


PAGE_TITLE = "Morocco Tourism Agent"
EXAMPLE_QUESTIONS = (
    "What are the main tourist attractions in Marrakech?",
    "Plan a two-day trip to Chefchaouen.",
    "Compare Marrakech and Chefchaouen.",
    "Estimate a three-day budget for Marrakech.",
    "How can I travel from Tangier to Chefchaouen?",
    "I prefer quiet cultural destinations with a moderate budget.",
    "Which city should I visit?",
)
SPECIALIZED_RESULTS = (
    "itinerary_result",
    "comparison_result",
    "budget_result",
    "transport_result",
)
ARCHITECTURE_PNG = PROJECT_ROOT / "artifacts" / "tourism_agent_graph.png"
ARCHITECTURE_MERMAID = (
    PROJECT_ROOT / "artifacts" / "tourism_agent_graph.mmd"
)


def _current_model() -> str:
    """Return the configured model name without reading environment values."""
    if LLM_PROVIDER == "ollama":
        return OLLAMA_CHAT_MODEL
    if LLM_PROVIDER == "gemini":
        return GEMINI_CHAT_MODEL
    return "Unsupported provider configuration"


def _initialize_session() -> None:
    """Create browser-session graph, thread, history, and result state."""
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph()
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"streamlit-{uuid4()}"
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def _start_new_conversation() -> None:
    """Reset only the current browser session's conversation state."""
    st.session_state.thread_id = f"streamlit-{uuid4()}"
    st.session_state.chat_messages = []
    st.session_state.last_result = None
    st.session_state.pending_question = None


def _exception_text(exc: BaseException) -> str:
    """Combine safe exception-chain messages for friendly classification."""
    messages: list[str] = []
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        messages.append(str(current))
        current = current.__cause__ or current.__context__
    return " ".join(messages).strip()


def _friendly_error(exc: BaseException) -> tuple[str, str | None]:
    """Map infrastructure failures to concise user-facing guidance."""
    detail = _exception_text(exc)
    normalized = detail.casefold()
    if (
        "model" in normalized
        and any(
            marker in normalized
            for marker in ("not found", "not installed", "pull model", "404")
        )
    ):
        return (
            f"The local Ollama model '{OLLAMA_CHAT_MODEL}' is not installed.",
            f"ollama pull {OLLAMA_CHAT_MODEL}",
        )
    if any(
        marker in normalized
        for marker in (
            "connection refused",
            "actively refused",
            "winerror 10061",
            "ollama service is unavailable",
            "connection refused by ollama",
        )
    ):
        return (
            "Ollama is not running or cannot be reached. Start Ollama and "
            "try again.",
            None,
        )
    if any(
        marker in normalized
        for marker in (
            "no vector database was found",
            "vector database was not found",
            "chroma database",
        )
    ):
        return (
            "The tourism vector database is missing.",
            "python -m scripts.index_documents",
        )
    if any(
        marker in normalized
        for marker in (
            "failed to retrieve tourism context",
            "retrieval test failed",
            "error embedding content",
        )
    ):
        return (
            "Tourism information could not be retrieved. Check the indexed "
            "documents and embedding configuration, then try again.",
            None,
        )
    if any(
        marker in normalized
        for marker in (
            "generation failed",
            "chat request failed",
            "answer generation",
            "gemini",
            "ollama",
        )
    ):
        return (
            "The chat model could not generate an answer. Verify the model "
            "service and configuration, then try again.",
            None,
        )
    return (
        "The tourism agent could not complete this request. Please try again.",
        None,
    )


def _render_agent_details(last_result: Mapping[str, object]) -> None:
    """Display routing, timing, memory, and specialized tool details."""
    state = last_result.get("state", {})
    if not isinstance(state, Mapping):
        return
    documents = state.get("retrieved_documents", [])
    document_count = len(documents) if isinstance(documents, Sequence) else 0

    with st.expander("Agent details"):
        detail_columns = st.columns(3)
        detail_columns[0].metric(
            "Detected intent",
            str(state.get("intent", "unknown")),
        )
        detail_columns[1].metric(
            "Selected path",
            str(state.get("selected_path", "unknown")),
        )
        detail_columns[2].metric(
            "Validation",
            str(state.get("validation_result", "unknown")),
        )
        st.write(
            f"**Response time:** "
            f"{float(last_result.get('response_time_seconds', 0.0)):.3f} s"
        )
        st.write(f"**Retrieved documents:** {document_count}")
        preferences = state.get("user_preferences", [])
        if isinstance(preferences, Sequence) and not isinstance(
            preferences,
            (str, bytes),
        ):
            preference_text = ", ".join(str(value) for value in preferences)
        else:
            preference_text = str(preferences or "")
        st.write(f"**Stored preferences:** {preference_text or 'None'}")
        st.write(f"**Thread ID:** `{st.session_state.thread_id}`")

        for field in SPECIALIZED_RESULTS:
            specialized_result = state.get(field)
            if not specialized_result:
                continue
            st.markdown(f"**{field.replace('_', ' ').title()}**")
            if isinstance(specialized_result, Mapping):
                st.json(dict(specialized_result), expanded=False)
            else:
                st.write(specialized_result)


def _document_preview(document: Document, limit: int = 280) -> str:
    """Return a short single-paragraph retrieved-content preview."""
    normalized = re.sub(r"\s+", " ", document.page_content).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _render_retrieved_sources(last_result: Mapping[str, object]) -> None:
    """Display unique source/page entries with compact previews."""
    state = last_result.get("state", {})
    if not isinstance(state, Mapping):
        return
    documents = state.get("retrieved_documents", [])
    if not isinstance(documents, Sequence):
        documents = []

    unique_documents: list[tuple[str, object, Document]] = []
    seen_sources: set[tuple[str, str]] = set()
    for document in documents:
        if not isinstance(document, Document):
            continue
        filename = str(document.metadata.get("filename", "unknown"))
        page = document.metadata.get("page", "unknown")
        source_key = (filename, str(page))
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        unique_documents.append((filename, page, document))

    with st.expander("Retrieved sources"):
        if not unique_documents:
            st.caption("No retrieved sources were returned for this answer.")
            return
        for filename, page, document in unique_documents:
            st.markdown(f"**{filename} — page {page}**")
            st.caption(_document_preview(document))


def _render_last_result() -> None:
    """Render the latest graph metadata and source evidence."""
    last_result = st.session_state.last_result
    if not isinstance(last_result, Mapping):
        return
    _render_agent_details(last_result)
    _render_retrieved_sources(last_result)


def _render_sidebar(provider: str, model: str) -> None:
    """Render project context, controls, examples, and graph artifacts."""
    with st.sidebar:
        st.title("🇲🇦 Morocco Tourism Agent")
        st.caption(f"Provider: **{provider}**")
        st.caption(f"Model: **{model}**")

        st.subheader("Conversation")
        if st.button(
            "New conversation",
            type="primary",
            width="stretch",
        ):
            _start_new_conversation()
            st.rerun()

        st.subheader("Example questions")
        for index, question in enumerate(EXAMPLE_QUESTIONS):
            if st.button(
                question,
                key=f"example-question-{index}",
                width="stretch",
            ):
                st.session_state.pending_question = question

        st.subheader("Architecture")
        st.caption(
            "Streamlit → LangGraph routing → retrieval and specialized "
            "tools → grounded generation → validation → thread memory"
        )
        if ARCHITECTURE_PNG.exists():
            with st.expander("Architecture diagram"):
                st.image(str(ARCHITECTURE_PNG), width="stretch")
        elif ARCHITECTURE_MERMAID.exists():
            with st.expander("Architecture diagram"):
                try:
                    mermaid_text = ARCHITECTURE_MERMAID.read_text(
                        encoding="utf-8"
                    )
                except OSError:
                    st.caption("The Mermaid graph could not be read.")
                else:
                    st.code(mermaid_text, language="mermaid")


def _render_chat_history() -> None:
    """Render the browser session's visible user and assistant messages."""
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _run_question(question: str) -> None:
    """Invoke the existing graph once and update session-visible results."""
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("assistant"):
        with st.spinner("Consulting the Morocco tourism knowledge base…"):
            started_at = time.perf_counter()
            try:
                result = st.session_state.graph.invoke(
                    {"question": question, "revision_count": 0},
                    config={
                        "configurable": {
                            "thread_id": st.session_state.thread_id,
                        }
                    },
                )
            except Exception as exc:
                message, command = _friendly_error(exc)
                st.error(message)
                if command:
                    st.code(f"Run: {command}")
                history_message = (
                    f"{message}\n\n`Run: {command}`" if command else message
                )
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": history_message}
                )
                st.session_state.last_result = None
                return
            response_time = time.perf_counter() - started_at

        answer = str(result.get("final_answer", "")).strip()
        if not answer:
            answer = (
                "The agent completed the request but did not return an "
                "answer. Please try rephrasing the question."
            )
        st.markdown(answer)

    st.session_state.chat_messages.append(
        {"role": "assistant", "content": answer}
    )
    st.session_state.last_result = {
        "state": result,
        "response_time_seconds": response_time,
    }
    _render_last_result()


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🇲🇦",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {max-width: 1100px; padding-top: 2rem;}
        [data-testid="stMetricValue"] {font-size: 1.15rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    provider = LLM_PROVIDER
    model = _current_model()
    try:
        _initialize_session()
    except Exception as exc:
        message, command = _friendly_error(exc)
        st.error(message)
        if command:
            st.code(f"Run: {command}")
        st.stop()

    _render_sidebar(provider, model)

    st.title("🇲🇦 Morocco Tourism Agent")
    st.write(
        "Ask grounded questions about Marrakech, Chefchaouen, itineraries, "
        "budgets, and transportation in Morocco."
    )
    status_columns = st.columns(2)
    status_columns[0].caption(f"Current LLM provider: **{provider}**")
    status_columns[1].caption(f"Current chat model: **{model}**")

    _render_chat_history()
    _render_last_result()

    pending_question = st.session_state.pending_question
    st.session_state.pending_question = None
    chat_question = st.chat_input("Ask about tourism in Morocco…")
    submitted_question = pending_question or chat_question
    if submitted_question is None:
        return
    question = str(submitted_question).strip()
    if not question:
        st.warning("Please enter a non-empty tourism question.")
        return
    _run_question(question)


if __name__ == "__main__":
    main()
