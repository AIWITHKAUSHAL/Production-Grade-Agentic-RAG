"""Embeddings components for the agentic RAG application."""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from .config import get_settings


class EURIEmbeddings(Embeddings):
    """OpenAI-compatible embeddings via EURI API with fallback to sentence-transformers."""

    def __init__(self):
        """Initialize lazy API and fallback embedding clients."""
        self.settings = get_settings()
        self._client = None
        self._fallback = None

    @property
    def client(self) -> OpenAI:
        """Initialize and return the EURI API client."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.settings.euri_api_key,
                base_url=self.settings.euri_base_url,
            )
        return self._client

    def _get_fallback(self):
        """Load the local sentence embedding model when available."""
        if self._fallback is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._fallback = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                pass
        return self._fallback

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts through EURI or the local fallback model."""
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.settings.euri_embedding_model,
            )
            return [item.embedding for item in response.data]
        except Exception:
            fb = self._get_fallback()
            if fb:
                return fb.encode(texts, normalize_embeddings=True).tolist()
            raise

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        return self.embed_documents([text])[0]


@lru_cache
def get_embeddings() -> EURIEmbeddings:
    """Return the shared EURI embedding provider."""
    return EURIEmbeddings()


def get_langchain_embeddings() -> OpenAIEmbeddings:
    """Create a LangChain embedding client for EURI."""
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.euri_embedding_model,
        openai_api_key=settings.euri_api_key,
        openai_api_base=settings.euri_base_url,
    )
