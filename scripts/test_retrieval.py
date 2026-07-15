"""Run a sample query against the persistent tourism vector database."""

import logging
import sys

from app.rag.retriever import retrieve_documents
from app.utils.logger import configure_application_logging


LOGGER = logging.getLogger(__name__)
SAMPLE_QUERY = "What are the main tourist attractions in Marrakech?"
PREVIEW_LENGTH = 500


def main() -> int:
    """Retrieve and print relevant chunks for the sample question."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    configure_application_logging()

    try:
        LOGGER.info("Query: %s", SAMPLE_QUERY)
        documents = retrieve_documents(SAMPLE_QUERY)
        if not documents:
            LOGGER.warning("No relevant documents were retrieved.")
            return 0

        for result_number, document in enumerate(documents, start=1):
            filename = document.metadata.get("filename", "unknown")
            page = document.metadata.get("page", "unknown")
            preview = document.page_content[:PREVIEW_LENGTH].strip()

            print(f"\n--- Result {result_number} ---")
            print(f"Filename: {filename}")
            print(f"Page: {page}")
            print(f"Content preview:\n{preview}")

        return 0
    except Exception as exc:
        LOGGER.error("Retrieval test failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
