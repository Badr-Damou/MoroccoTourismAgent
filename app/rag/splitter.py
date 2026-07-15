"""Split loaded tourism documents into retrieval-ready text chunks."""

from collections.abc import Sequence
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def split_documents(documents: Sequence[Document]) -> list[Document]:
    """Split documents and assign a stable, unique ID to every chunk."""
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(list(documents))

    for chunk_index, chunk in enumerate(chunks):
        filename = str(chunk.metadata.get("filename", "unknown"))
        page = str(chunk.metadata.get("page", "unknown"))
        start_index = str(chunk.metadata.get("start_index", "unknown"))
        identity = "|".join(
            (
                filename,
                page,
                start_index,
                str(chunk_index),
                chunk.page_content,
            )
        )
        chunk.metadata["chunk_id"] = str(uuid5(NAMESPACE_URL, identity))

    return chunks
