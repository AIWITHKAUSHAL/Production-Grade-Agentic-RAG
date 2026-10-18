"""
Self-RAG: The model decides when to retrieve and evaluates its own outputs.
Implements: IsREL (relevance), IsSUP (support), IsUSE (utility) tokens.
"""

import json
import logging

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from ..core.llm import get_grader_llm

logger = logging.getLogger(__name__)


class SelfRAGController:
    """Controls self-RAG flow: decide retrieve → generate → reflect."""

    RETRIEVE_DECISION_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Decide if retrieval is needed to answer this query.
Return JSON: {{"retrieve": true/false, "reason": "brief reason"}}
Retrieve=true if the query needs factual information not in general knowledge.""",
            ),
            ("human", "Query: {query}"),
        ]
    )

    ISREL_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Check if this document passage is relevant to answering the query.
Return JSON: {{"relevant": true/false, "score": 0.0-1.0}}""",
            ),
            ("human", "Query: {query}\nPassage: {passage}"),
        ]
    )

    ISSUP_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Check if the generated response is fully supported by the retrieved passages.
Return JSON: {{"supported": true/false, "score": 0.0-1.0, "unsupported": "any unsupported claim"}}""",
            ),
            ("human", "Passages:\n{passages}\n\nResponse: {response}"),
        ]
    )

    ISUSE_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Rate the utility of this response for answering the original query (1-5).
Return JSON: {{"utility": 1-5, "reason": "brief"}}""",
            ),
            ("human", "Query: {query}\nResponse: {response}"),
        ]
    )

    def __init__(self):
        """Initialize the SelfRAGController dependencies and state."""
        self.llm = get_grader_llm()

    async def should_retrieve(self, query: str) -> tuple[bool, str]:
        """Decide whether a query needs external document retrieval."""
        chain = self.RETRIEVE_DECISION_PROMPT | self.llm
        try:
            result = await chain.ainvoke({"query": query})
            text = result.content.strip()
            start, end = text.find("{"), text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return bool(data.get("retrieve", True)), str(data.get("reason", ""))
        except Exception as e:
            logger.error(f"Self-RAG retrieve decision failed: {e}")
        return True, "defaulting to retrieval"

    async def is_relevant(self, query: str, passage: str) -> tuple[bool, float]:
        """Score whether a passage helps answer the query."""
        chain = self.ISREL_PROMPT | self.llm
        try:
            result = await chain.ainvoke({"query": query, "passage": passage[:600]})
            text = result.content.strip()
            start, end = text.find("{"), text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return bool(data.get("relevant", False)), float(data.get("score", 0.5))
        except Exception as e:
            logger.error(f"IsREL check failed: {e}")
        return False, 0.0

    async def is_supported(self, passages: list[str], response: str) -> tuple[bool, float, str]:
        """Check whether retrieved passages support a generated answer."""
        combined = "\n---\n".join(passages[:3])
        chain = self.ISSUP_PROMPT | self.llm
        try:
            result = await chain.ainvoke({"passages": combined[:1200], "response": response[:600]})
            text = result.content.strip()
            start, end = text.find("{"), text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return (
                    bool(data.get("supported", False)),
                    float(data.get("score", 0.5)),
                    str(data.get("unsupported", "")),
                )
        except Exception as e:
            logger.error(f"IsSUP check failed: {e}")
        return False, 0.0, "check failed"

    async def is_useful(self, query: str, response: str) -> tuple[int, str]:
        """Rate how well a response answers the original query."""
        chain = self.ISUSE_PROMPT | self.llm
        try:
            result = await chain.ainvoke({"query": query, "response": response[:600]})
            text = result.content.strip()
            start, end = text.find("{"), text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return int(data.get("utility", 3)), str(data.get("reason", ""))
        except Exception as e:
            logger.error(f"IsUSE check failed: {e}")
        return 3, "check failed"

    async def filter_relevant_docs(self, query: str, documents: list[Document]) -> list[Document]:
        """Keep documents that pass the relevance check."""
        relevant = []
        for doc in documents:
            is_rel, score = await self.is_relevant(query, doc.page_content)
            if is_rel:
                enriched = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "self_rag_rel_score": score},
                )
                relevant.append(enriched)
        return relevant
