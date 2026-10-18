"""
Corrective RAG (CRAG): If retrieved docs are not relevant, fall back to web search.
"""

import logging

from langchain_core.documents import Document

from .grader import DocumentGrader

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.5


class CRAGEvaluator:
    """Evaluate retrieval quality and decide corrective action."""

    def __init__(self):
        """Initialize the CRAGEvaluator dependencies and state."""
        self.grader = DocumentGrader()

    async def evaluate(self, query: str, documents: list[Document]) -> tuple[str, list[Document]]:
        """
        Returns: (action, filtered_docs)
        action: "use" | "web_search" | "hybrid"
        """
        if not documents:
            return "web_search", []

        graded = []
        relevant_count = 0
        for doc in documents:
            relevant, score, reason = await self.grader.grade(query, doc)
            enriched = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "relevance_score": score, "relevance_reason": reason},
            )
            graded.append(enriched)
            if relevant:
                relevant_count += 1

        ratio = relevant_count / len(documents)
        logger.info(f"CRAG: {relevant_count}/{len(documents)} docs relevant (ratio={ratio:.2f})")

        if ratio >= 0.6:
            # Most docs relevant — use them
            relevant_docs = [
                d for d in graded if d.metadata.get("relevance_score", 0) >= RELEVANCE_THRESHOLD
            ]
            return "use", relevant_docs
        elif ratio >= 0.2:
            # Some relevant — supplement with web search
            relevant_docs = [
                d for d in graded if d.metadata.get("relevance_score", 0) >= RELEVANCE_THRESHOLD
            ]
            return "hybrid", relevant_docs
        else:
            # Mostly irrelevant — go to web
            return "web_search", []

    async def strip_irrelevant(self, query: str, documents: list[Document]) -> list[Document]:
        """Strip irrelevant sentences from documents."""
        result = []
        for doc in documents:
            sentences = [s.strip() for s in doc.page_content.split(".") if s.strip()]
            relevant_sentences = []
            query_words = set(query.lower().split())
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words & sentence_words)
                if overlap >= 1 or len(relevant_sentences) < 3:
                    relevant_sentences.append(sentence)
            if relevant_sentences:
                new_content = ". ".join(relevant_sentences[:10]) + "."
                result.append(
                    Document(
                        page_content=new_content, metadata={**doc.metadata, "crag_stripped": True}
                    )
                )
        return result
