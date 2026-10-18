"""Rrf components for the agentic RAG application."""

from langchain_core.documents import Document


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int = 60,
    weights: list[float] = None,
) -> list[Document]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.
    k: constant that controls ranking impact (default 60 per original RRF paper)
    weights: per-list weights (uniform if None)
    """
    if not result_lists:
        return []

    if weights is None:
        weights = [1.0] * len(result_lists)

    # Map doc_id → (document, score)
    scores: dict[str, tuple[Document, float]] = {}

    for rank_list, weight in zip(result_lists, weights):
        for rank, doc in enumerate(rank_list):
            doc_id = _doc_id(doc)
            rrf_score = weight / (k + rank + 1)
            if doc_id in scores:
                scores[doc_id] = (scores[doc_id][0], scores[doc_id][1] + rrf_score)
            else:
                scores[doc_id] = (doc, rrf_score)

    sorted_docs = sorted(scores.values(), key=lambda x: x[1], reverse=True)
    result = []
    for doc, score in sorted_docs:
        enriched = Document(
            page_content=doc.page_content, metadata={**doc.metadata, "rrf_score": round(score, 6)}
        )
        result.append(enriched)
    return result


def _doc_id(doc: Document) -> str:
    """Derive a document identity for merging ranked result lists."""
    meta = doc.metadata
    if "doc_id" in meta:
        return str(meta["doc_id"])
    return f"{meta.get('source', '')}__{meta.get('page', '')}__{hash(doc.page_content[:100])}"
