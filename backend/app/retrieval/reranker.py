import logging
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from ..core.llm import get_grader_llm

logger = logging.getLogger(__name__)

_cross_encoder = None


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            logger.warning(f"Cross-encoder unavailable: {e}")
    return _cross_encoder


class CrossEncoderReranker:
    """Rerank with a local cross-encoder model."""

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        ce = get_cross_encoder()
        if ce is None or not documents:
            return documents[:top_k]
        pairs = [[query, doc.page_content] for doc in documents]
        try:
            scores = ce.predict(pairs)
            scored = list(zip(documents, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            result = []
            for doc, score in scored[:top_k]:
                enriched = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "cross_encoder_score": float(score)}
                )
                result.append(enriched)
            return result
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return documents[:top_k]


class LLMReranker:
    """Rerank documents using the LLM for relevance scoring."""

    RERANK_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a relevance scorer. Given a query and a document excerpt,
score the relevance from 0 to 10 (10 = highly relevant, 0 = irrelevant).
Return only a JSON object: {{"score": <number>, "reason": "<brief reason>"}}"""),
        ("human", "Query: {query}\n\nDocument: {document}\n\nRelevance score (0-10):"),
    ])

    def __init__(self):
        self.llm = get_grader_llm()

    async def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        if not documents:
            return []
        import json
        import asyncio

        async def score_doc(doc: Document) -> Tuple[Document, float]:
            try:
                chain = self.RERANK_PROMPT | self.llm
                result = await chain.ainvoke({
                    "query": query,
                    "document": doc.page_content[:500],
                })
                text = result.content.strip()
                # Extract JSON
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    data = json.loads(text[start:end])
                    score = float(data.get("score", 5))
                    reason = data.get("reason", "")
                else:
                    score = 5.0
                    reason = ""
                enriched = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "llm_rerank_score": score, "llm_rerank_reason": reason}
                )
                return enriched, score
            except Exception:
                return doc, 5.0

        tasks = [score_doc(doc) for doc in documents]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in results[:top_k]]


def get_reranker(method: str = "cross_encoder") -> "CrossEncoderReranker | LLMReranker":
    if method == "llm":
        return LLMReranker()
    return CrossEncoderReranker()
