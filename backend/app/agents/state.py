from typing import TypedDict, List, Optional, Dict, Any, Annotated
from langchain_core.documents import Document
import operator


class AgentState(TypedDict):
    # Core query
    query: str
    rewritten_query: Optional[str]

    # Retrieval
    documents: List[Document]
    graded_documents: List[Document]
    web_search_results: List[Document]

    # Generation
    generation: Optional[str]
    sources: List[str]

    # Grading scores
    hallucination_score: float
    answer_relevance_score: float
    grounding_score: float

    # Control flow
    retry_count: int
    route_decision: str           # "vectorstore" | "web_search" | "general" | "calculator"
    crag_decision: str            # "use" | "web_search" | "hybrid"
    should_retrieve: bool
    generation_quality: str       # "good" | "poor"

    # Self-RAG reflection tokens
    self_rag_isrel: bool
    self_rag_issup: bool
    self_rag_isuse: int

    # Final answer
    final_answer: Optional[str]
    error: Optional[str]

    # Tool results
    calculator_result: Optional[str]

    # Execution trace (for UI)
    steps: Annotated[List[Dict[str, Any]], operator.add]

    # Metadata
    metadata: Dict[str, Any]
    session_id: Optional[str]
