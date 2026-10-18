"""Grader components for the agentic RAG application."""

import json
import logging

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from ..core.llm import get_grader_llm

logger = logging.getLogger(__name__)


class DocumentGrader:
    """Grade document relevance to a query using LLM."""

    GRADE_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a document relevance grader. Assess whether the document is relevant
to the query. Return JSON with:
{{"relevant": true/false, "score": 0.0-1.0, "reason": "brief reason"}}
Relevant means the document contains information useful for answering the query.""",
            ),
            ("human", "Query: {query}\n\nDocument: {document}\n\nGrade:"),
        ]
    )

    def __init__(self):
        """Initialize the DocumentGrader dependencies and state."""
        self.llm = get_grader_llm()

    async def grade(self, query: str, document: Document) -> tuple[bool, float, str]:
        """Score whether a document is relevant to a query."""
        chain = self.GRADE_PROMPT | self.llm
        try:
            result = await chain.ainvoke(
                {
                    "query": query,
                    "document": document.page_content[:800],
                }
            )
            text = result.content.strip()
            start, end = text.find("{"), text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return (
                    bool(data.get("relevant", False)),
                    float(data.get("score", 0.5)),
                    str(data.get("reason", "")),
                )
        except Exception as e:
            logger.error(f"Document grading failed: {e}")
        return False, 0.0, "grading failed"

    async def grade_all(self, query: str, documents: list[Document]) -> list[Document]:
        """Return relevant documents enriched with grading scores."""
        graded = []
        for doc in documents:
            relevant, score, reason = await self.grade(query, doc)
            if relevant:
                enriched = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "relevance_score": score,
                        "relevance_reason": reason,
                    },
                )
                graded.append(enriched)
        return graded


class HallucinationGrader:
    """Check if a generation is grounded in the provided documents."""

    HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a grounding verifier. Check if the answer is supported by the documents.
Return JSON: {{"grounded": true/false, "score": 0.0-1.0, "unsupported_claims": ["claim1", ...]}}
grounded=true means all factual claims are supported by the documents.""",
            ),
            ("human", "Documents:\n{documents}\n\nAnswer: {answer}\n\nIs the answer grounded?"),
        ]
    )

    def __init__(self):
        """Initialize the HallucinationGrader dependencies and state."""
        self.llm = get_grader_llm()

    async def grade(self, answer: str, documents: list[Document]) -> tuple[bool, float, list[str]]:
        """Check answer grounding and identify unsupported claims."""
        if not documents:
            return False, 0.0, ["no documents provided"]
        doc_text = "\n\n---\n\n".join(
            f"[Doc {i + 1}]: {d.page_content[:400]}" for i, d in enumerate(documents[:5])
        )
        chain = self.HALLUCINATION_PROMPT | self.llm
        try:
            result = await chain.ainvoke({"documents": doc_text, "answer": answer[:600]})
            text = result.content.strip()
            start, end = text.find("{"), text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return (
                    bool(data.get("grounded", False)),
                    float(data.get("score", 0.5)),
                    data.get("unsupported_claims", []),
                )
        except Exception as e:
            logger.error(f"Hallucination grading failed: {e}")
        return False, 0.0, ["grading failed"]


class AnswerRelevanceGrader:
    """Grade whether the answer actually addresses the question."""

    RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an answer relevance grader. Check if the answer addresses the question.
Return JSON: {{"relevant": true/false, "score": 0.0-1.0, "reason": "brief"}}""",
            ),
            (
                "human",
                "Question: {question}\n\nAnswer: {answer}\n\nDoes the answer address the question?",
            ),
        ]
    )

    def __init__(self):
        """Initialize the AnswerRelevanceGrader dependencies and state."""
        self.llm = get_grader_llm()

    async def grade(self, question: str, answer: str) -> tuple[bool, float, str]:
        """Score whether an answer addresses the question."""
        chain = self.RELEVANCE_PROMPT | self.llm
        try:
            result = await chain.ainvoke({"question": question, "answer": answer[:600]})
            text = result.content.strip()
            start, end = text.find("{"), text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return (
                    bool(data.get("relevant", False)),
                    float(data.get("score", 0.5)),
                    str(data.get("reason", "")),
                )
        except Exception as e:
            logger.error(f"Answer relevance grading failed: {e}")
        return False, 0.0, "grading failed"
