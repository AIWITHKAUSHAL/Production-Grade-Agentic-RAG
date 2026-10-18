"""State components for the agentic RAG application."""

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    # Core query
    """Shared graph state for retrieval, generation, and evaluation."""

    query: str
    rewritten_query: str | None

    # Retrieval
    documents: list[Document]
    graded_documents: list[Document]
    web_search_results: list[Document]

    # Generation
    generation: str | None
    sources: list[str]

    # Grading scores
    hallucination_score: float
    answer_relevance_score: float
    grounding_score: float

    # Control flow
    retry_count: int
    route_decision: str  # "vectorstore" | "web_search" | "general" | "calculator"
    crag_decision: str  # "use" | "web_search" | "hybrid"
    should_retrieve: bool
    generation_quality: str  # "good" | "poor"

    # Self-RAG reflection tokens
    self_rag_isrel: bool
    self_rag_issup: bool
    self_rag_isuse: int

    # Final answer
    final_answer: str | None
    error: str | None

    # Tool results
    calculator_result: str | None

    # Execution trace (for UI)
    steps: Annotated[list[dict[str, Any]], operator.add]

    # Metadata
    metadata: dict[str, Any]
    session_id: str | None
