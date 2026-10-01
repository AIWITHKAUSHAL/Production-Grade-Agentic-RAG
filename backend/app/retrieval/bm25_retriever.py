import re
import logging
import asyncio
from typing import List, Optional
from dataclasses import dataclass, field
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    STOPWORDS = {"a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
                 "of", "and", "or", "but", "with", "this", "that", "are", "was"}
    tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class BM25Index:
    """In-memory BM25 index for keyword search with persistent backing."""

    def __init__(self):
        self._documents: List[Document] = []
        self._tokenized: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._lock = asyncio.Lock()

    async def add_documents(self, docs: List[Document]) -> None:
        async with self._lock:
            for doc in docs:
                self._documents.append(doc)
                self._tokenized.append(tokenize(doc.page_content))
            self._rebuild()

    def _rebuild(self):
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)

    async def search(self, query: str, k: int = 5) -> List[Document]:
        if not self._bm25 or not self._documents:
            return []
        async with self._lock:
            query_tokens = tokenize(query)
            if not query_tokens:
                return []
            scores = self._bm25.get_scores(query_tokens)
            top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            results = []
            for idx in top_k_idx:
                if scores[idx] > 0:
                    doc = self._documents[idx]
                    doc_copy = Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, "bm25_score": float(scores[idx])}
                    )
                    results.append(doc_copy)
            return results

    @property
    def doc_count(self) -> int:
        return len(self._documents)


# Global BM25 instance
_bm25_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index
