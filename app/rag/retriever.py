"""Retrieve semantically relevant tourism chunks from ChromaDB."""

from langchain_core.documents import Document

from app.rag.vectordb import load_vector_store


def retrieve_documents(
    query: str,
    number_of_results: int = 4,
) -> list[Document]:
    """Return the chunks most semantically similar to a user query."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("The retrieval query cannot be empty.")
    if number_of_results < 1:
        raise ValueError("number_of_results must be at least 1.")

    vector_store = load_vector_store()
    return vector_store.similarity_search(
        query=normalized_query,
        k=number_of_results,
    )
