"""Create and reopen the persistent Chroma vector database."""

from collections.abc import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.rag.embeddings import get_embedding_model
from app.utils.config import CHROMA_COLLECTION_NAME, VECTOR_DB_DIR


class VectorStoreError(RuntimeError):
    """Indicate that the persistent vector store is unavailable or invalid."""


def _new_vector_store() -> Chroma:
    """Create a Chroma client configured for local persistence."""
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=str(VECTOR_DB_DIR),
    )


def create_vector_store(chunks: Sequence[Document]) -> Chroma:
    """Replace the persistent collection with the supplied document chunks.

    Replacing the collection makes indexing repeatable and prevents duplicate or
    stale chunks from accumulating across indexing runs.
    """
    documents = list(chunks)
    if not documents:
        raise ValueError("Cannot create a vector store without document chunks.")

    chunk_ids = [str(chunk.metadata.get("chunk_id", "")) for chunk in documents]
    if any(not chunk_id for chunk_id in chunk_ids):
        raise ValueError("Every document chunk must contain a chunk_id.")
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Document chunk_id values must be unique.")

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    vector_store = _new_vector_store()

    try:
        vector_store.delete_collection()
        vector_store = _new_vector_store()
        vector_store.add_documents(documents=documents, ids=chunk_ids)
    except Exception as exc:
        raise VectorStoreError("Failed to create the Chroma vector store.") from exc

    return vector_store


def load_vector_store() -> Chroma:
    """Reopen and validate the existing persistent Chroma collection."""
    if not VECTOR_DB_DIR.exists() or not any(VECTOR_DB_DIR.iterdir()):
        raise VectorStoreError(
            "No vector database was found. Run "
            "`python -m scripts.index_documents` first."
        )

    try:
        vector_store = _new_vector_store()
        stored_documents = vector_store.get(limit=1, include=[])
    except Exception as exc:
        raise VectorStoreError("Failed to load the Chroma vector store.") from exc

    if not stored_documents.get("ids"):
        raise VectorStoreError(
            "The vector database is empty. Run "
            "`python -m scripts.index_documents` first."
        )

    return vector_store
