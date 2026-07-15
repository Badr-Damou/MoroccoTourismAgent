"""Expose the existing tourism retriever through a LangChain tool."""

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.tools import tool

from app.rag.retriever import retrieve_documents


NO_DOCUMENTS_MESSAGE = "No relevant tourism documents were found."


def format_tourism_documents(documents: Sequence[Document]) -> str:
    """Format retrieved chunks with source metadata for grounded generation."""
    if not documents:
        return NO_DOCUMENTS_MESSAGE

    formatted_chunks: list[str] = []
    for result_number, document in enumerate(documents, start=1):
        filename = document.metadata.get("filename", "unknown")
        page = document.metadata.get("page", "unknown")
        formatted_chunks.append(
            f"[Source {result_number}: {filename}, page {page}]\n"
            f"{document.page_content.strip()}"
        )

    return "\n\n".join(formatted_chunks)


@tool
def search_tourism_documents(query: str) -> str:
    """Search the Morocco tourism documents for four relevant chunks."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("The tourism search query cannot be empty.")

    try:
        documents = retrieve_documents(
            normalized_query,
            number_of_results=4,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to search the tourism documents: {exc}"
        ) from exc

    return format_tourism_documents(documents)
