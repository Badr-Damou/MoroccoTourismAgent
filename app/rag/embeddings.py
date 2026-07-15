"""Create the OpenAI embedding model used by the RAG pipeline."""

from langchain_openai import OpenAIEmbeddings

from app.utils.config import get_openai_api_key


EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding_model() -> OpenAIEmbeddings:
    """Return a configured, reusable OpenAI embeddings client."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=get_openai_api_key(),
    )
