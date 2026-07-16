"""Define the typed state passed through the tourism-agent graph."""

from typing import Annotated

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class TourismAgentState(TypedDict, total=False):
    """Represent data produced while answering one tourism question."""

    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    intent: str
    retrieved_documents: list[Document]
    context: str
    user_preferences: list[str]
    selected_path: str
    itinerary_result: dict[str, object] | str
    comparison_result: dict[str, object] | str
    budget_result: dict[str, object] | str
    transport_result: dict[str, object] | str
    final_answer: str
    validation_result: str
    validation_feedback: str
    revision_count: int
