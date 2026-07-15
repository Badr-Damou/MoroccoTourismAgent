"""Load PDF tourism documents from the configured document directory."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from app.utils.config import DOCUMENTS_DIR


class DocumentLoadingError(RuntimeError):
    """Indicate that a PDF could not be loaded successfully."""


def find_pdf_files(directory: Path = DOCUMENTS_DIR) -> list[Path]:
    """Return all PDF files below ``directory`` in deterministic order."""
    if not directory.exists():
        return []

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def load_pdf_documents(directory: Path = DOCUMENTS_DIR) -> list[Document]:
    """Load every PDF page and attach normalized source metadata.

    Args:
        directory: Directory to scan recursively for PDF files.

    Returns:
        One LangChain document per loaded PDF page.

    Raises:
        DocumentLoadingError: If any discovered PDF cannot be parsed.
    """
    documents: list[Document] = []

    for pdf_path in find_pdf_files(directory):
        try:
            pages = PyPDFLoader(str(pdf_path)).load()
        except Exception as exc:
            raise DocumentLoadingError(
                f"Failed to load PDF document: {pdf_path}"
            ) from exc

        for page_index, document in enumerate(pages):
            document.metadata.update(
                {
                    "filename": pdf_path.name,
                    "page": document.metadata.get("page", page_index),
                    "file_type": "pdf",
                }
            )
            documents.append(document)

    return documents
