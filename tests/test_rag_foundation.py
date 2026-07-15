"""Test the RAG foundation without making external API requests."""

import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pypdf import PdfWriter

from app.rag.loader import load_pdf_documents
from app.rag.splitter import split_documents
from app.rag import vectordb
from app.utils.config import PROJECT_ROOT


RUNTIME_DIR = PROJECT_ROOT / "tests" / ".runtime"
PDF_TEST_DIR = RUNTIME_DIR / "documents"
VECTOR_TEST_DIR = RUNTIME_DIR / "vectordb"


class KeywordEmbeddings(Embeddings):
    """Provide deterministic vectors for local Chroma integration tests."""

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized_text = text.lower()
        return [
            float("marrakech" in normalized_text),
            float("essaouira" in normalized_text),
            min(len(normalized_text) / 1000, 1.0),
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document strings deterministically."""
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string deterministically."""
        return self._embed(text)


class RagFoundationTests(unittest.TestCase):
    """Cover PDF loading, splitting, persistence, and retrieval boundaries."""

    def test_pdf_loader_adds_required_metadata(self) -> None:
        """Ensure PDF pages retain normalized source metadata."""
        PDF_TEST_DIR.mkdir(parents=True, exist_ok=True)
        pdf_path = PDF_TEST_DIR / "tourism.PDF"
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with pdf_path.open("wb") as pdf_file:
            writer.write(pdf_file)

        documents = load_pdf_documents(PDF_TEST_DIR)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata["filename"], "tourism.PDF")
        self.assertEqual(documents[0].metadata["page"], 0)
        self.assertEqual(documents[0].metadata["file_type"], "pdf")

    def test_splitter_assigns_unique_chunk_ids(self) -> None:
        """Ensure each split receives a non-empty and unique chunk ID."""
        source = Document(
            page_content=("Morocco tourism information. " * 100),
            metadata={"filename": "guide.pdf", "page": 0},
        )

        chunks = split_documents([source])
        chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]

        self.assertGreater(len(chunks), 1)
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
        self.assertTrue(all(chunk_ids))

    def test_chroma_store_persists_and_retrieves_documents(self) -> None:
        """Ensure a Chroma collection can be created, reopened, and queried."""
        documents = split_documents(
            [
                Document(
                    page_content="Marrakech has Jemaa el-Fnaa and Bahia Palace.",
                    metadata={"filename": "marrakech.pdf", "page": 0},
                ),
                Document(
                    page_content="Essaouira is known for its Atlantic coast.",
                    metadata={"filename": "essaouira.pdf", "page": 0},
                ),
            ]
        )

        VECTOR_TEST_DIR.mkdir(parents=True, exist_ok=True)
        with (
            patch.object(vectordb, "VECTOR_DB_DIR", VECTOR_TEST_DIR),
            patch.object(
                vectordb,
                "get_embedding_model",
                return_value=KeywordEmbeddings(),
            ),
        ):
            created_store = vectordb.create_vector_store(documents)
            loaded_store = vectordb.load_vector_store()
            results = loaded_store.similarity_search("Marrakech", k=1)

            self.assertEqual(
                created_store.get()["ids"],
                loaded_store.get()["ids"],
            )
            self.assertEqual(results[0].metadata["filename"], "marrakech.pdf")


if __name__ == "__main__":
    unittest.main()
