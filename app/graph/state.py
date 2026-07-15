"""Define the typed state passed through the tourism-agent graph."""

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class TourismAgentState(TypedDict, total=False):
    """Represent data produced while answering one tourism question."""

    messages: list[BaseMessage]
    question: str
    intent: str
    retrieved_documents: list[Document]
    context: str
    final_answer: str
    validation_result: str
    revision_count: int
