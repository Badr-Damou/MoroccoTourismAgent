"""Index local PDF tourism documents into the persistent Chroma database."""

import logging

from app.rag.loader import load_pdf_documents
from app.rag.splitter import split_documents
from app.rag.vectordb import create_vector_store
from app.utils.logger import configure_application_logging


LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Run the complete PDF loading, chunking, and indexing pipeline."""
    configure_application_logging()

    try:
        LOGGER.info("Loading PDF documents...")
        documents = load_pdf_documents()
        if not documents:
            raise FileNotFoundError(
                "No PDF documents were found in data/documents/."
            )
        LOGGER.info("Loaded %d PDF pages.", len(documents))

        LOGGER.info("Splitting pages into chunks...")
        chunks = split_documents(documents)
        if not chunks:
            raise ValueError("The loaded PDFs did not contain indexable text.")
        LOGGER.info("Created %d chunks.", len(chunks))

        LOGGER.info("Generating embeddings and writing ChromaDB...")
        create_vector_store(chunks)
        LOGGER.info("Indexing completed successfully.")
        return 0
    except Exception as exc:
        LOGGER.error("Indexing failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
