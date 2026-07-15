"""Create the Gemini embedding model used by the RAG pipeline."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.utils.config import get_google_api_key


EMBEDDING_MODEL = "models/gemini-embedding-001"


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Return a configured Gemini embeddings client for the RAG pipeline."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=get_google_api_key(),
    )
