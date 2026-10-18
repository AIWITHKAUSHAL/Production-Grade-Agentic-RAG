"""Hybrid search components for the agentic RAG application."""

import logging

from langchain_core.documents import Document

from .bm25_retriever import get_bm25_index
from .reranker import CrossEncoderReranker
from .rrf import reciprocal_rank_fusion
from .vector_store import similarity_search_with_score

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    Hybrid search combining semantic (vector) + keyword (BM25) retrieval
    with Reciprocal Rank Fusion and optional cross-encoder reranking.
    """

    def __init__(self, alpha: float = 0.5, k: int = 5, rerank: bool = True, rerank_top_k: int = 3):
        """Initialize the HybridSearcher dependencies and state."""
        self.alpha = alpha  # 0=pure BM25, 1=pure semantic
        self.k = k
        self.rerank = rerank
        self.rerank_top_k = rerank_top_k
        self._reranker = CrossEncoderReranker() if rerank else None

    async def search(self, query: str, filter_metadata: dict | None = None) -> list[Document]:
        """Fuse semantic and keyword matches, then optionally rerank."""
        fetch_k = max(self.k * 2, 10)

        # Semantic search
        semantic_docs = []
        try:
            results = await similarity_search_with_score(query, k=fetch_k)
            semantic_docs = [
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "semantic_score": round(float(score), 4)},
                )
                for doc, score in results
            ]
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")

        # BM25 keyword search
        bm25_docs = []
        try:
            bm25_index = get_bm25_index()
            bm25_docs = await bm25_index.search(query, k=fetch_k)
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")

        if not semantic_docs and not bm25_docs:
            return []

        # Combine with alpha-weighted RRF
        semantic_weight = self.alpha
        bm25_weight = 1.0 - self.alpha

        result_lists = []
        weights = []
        if semantic_docs:
            result_lists.append(semantic_docs)
            weights.append(semantic_weight)
        if bm25_docs:
            result_lists.append(bm25_docs)
            weights.append(bm25_weight)

        fused = reciprocal_rank_fusion(result_lists, weights=weights)
        candidates = fused[:fetch_k]

        # Cross-encoder reranking
        if self.rerank and self._reranker and candidates:
            candidates = self._reranker.rerank(query, candidates, top_k=self.rerank_top_k)
        else:
            candidates = candidates[: self.k]

        return candidates


_searcher: HybridSearcher | None = None


def get_hybrid_searcher(
    alpha: float = 0.5, k: int = 5, rerank: bool = True, rerank_top_k: int = 3
) -> HybridSearcher:
    """Return the shared hybrid retrieval service."""
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher(alpha=alpha, k=k, rerank=rerank, rerank_top_k=rerank_top_k)
    return _searcher
