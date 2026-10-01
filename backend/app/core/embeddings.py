from functools import lru_cache
from typing import List
import numpy as np
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from .config import get_settings


class EURIEmbeddings(Embeddings):
    """OpenAI-compatible embeddings via EURI API with fallback to sentence-transformers."""

    def __init__(self):
        self.settings = get_settings()
        self._client = None
        self._fallback = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.settings.euri_api_key,
                base_url=self.settings.euri_base_url,
            )
        return self._client

    def _get_fallback(self):
        if self._fallback is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._fallback = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                pass
        return self._fallback

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
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

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


@lru_cache()
def get_embeddings() -> EURIEmbeddings:
    return EURIEmbeddings()


def get_langchain_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.euri_embedding_model,
        openai_api_key=settings.euri_api_key,
        openai_api_base=settings.euri_base_url,
    )
