"""Graph components for the agentic RAG application."""

import logging
import os
from typing import Literal

from langgraph.graph import END, START, StateGraph

from ..core.config import get_settings
from .nodes import (
    calculator_node,
    evaluate_generation,
    final_answer,
    generate,
    grade_documents,
    retrieve,
    rewrite_query,
    route_query,
    understand_query,
    web_search_node,
)
from .state import AgentState

logger = logging.getLogger(__name__)


# ─── Conditional Edge: Route Decision ────────────────────────────────────────
def decide_route(state: AgentState) -> Literal["retrieve", "web_search", "calculator", "generate"]:
    """Select the next graph node from the routing decision."""
    route = state.get("route_decision", "vectorstore")
    if route == "calculator":
        return "calculator"
    elif route == "web_search":
        return "web_search"
    elif route == "general":
        return "generate"
    else:
        return "retrieve"


# ─── Conditional Edge: After Grade Documents ─────────────────────────────────
def decide_after_grade(state: AgentState) -> Literal["generate", "rewrite_query"]:
    """Generate from relevant documents or rewrite the query."""
    docs = state.get("graded_documents", [])
    retry = state.get("retry_count", 0)
    settings = get_settings()

    if not docs and retry < settings.max_retries:
        return "rewrite_query"
    return "generate"


# ─── Conditional Edge: After Evaluate ────────────────────────────────────────
def decide_after_evaluate(state: AgentState) -> Literal["final_answer", "rewrite_query"]:
    """Finish an acceptable answer or retry within the limit."""
    quality = state.get("generation_quality", "poor")
    retry = state.get("retry_count", 0)
    settings = get_settings()

    if quality == "good" or retry >= settings.max_retries:
        return "final_answer"
    return "rewrite_query"


# ─── Conditional Edge: After Rewrite ─────────────────────────────────────────
def decide_after_rewrite(state: AgentState) -> Literal["retrieve", "web_search"]:
    """Resume retrieval using the existing search route."""
    route = state.get("route_decision", "vectorstore")
    if route == "web_search":
        return "web_search"
    return "retrieve"


def build_graph():
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("understand_query", understand_query)
    graph.add_node("route", route_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("web_search", web_search_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("evaluate", evaluate_generation)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("final_answer", final_answer)

    # Entry point
    graph.add_edge(START, "understand_query")
    graph.add_edge("understand_query", "route")

    # Route conditional
    graph.add_conditional_edges(
        "route",
        decide_route,
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
            "calculator": "calculator",
            "generate": "generate",
        },
    )

    # After retrieval → grade
    graph.add_edge("retrieve", "grade_documents")
    graph.add_edge("web_search", "grade_documents")

    # After grade → decide
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grade,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )

    # After generate → evaluate
    graph.add_edge("generate", "evaluate")

    # After evaluate → decide
    graph.add_conditional_edges(
        "evaluate",
        decide_after_evaluate,
        {
            "final_answer": "final_answer",
            "rewrite_query": "rewrite_query",
        },
    )

    # After rewrite → retrieve or web
    graph.add_conditional_edges(
        "rewrite_query",
        decide_after_rewrite,
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
        },
    )

    # Calculator goes straight to end (already sets final_answer)
    graph.add_edge("calculator", "final_answer")
    graph.add_edge("final_answer", END)

    return graph


def compile_graph(checkpointer=None):
    """Compile the graph with optional checkpointing."""
    graph = build_graph()

    # Setup LangSmith tracing
    settings = get_settings()
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


# Global compiled graph
_compiled_graph = None


def get_graph():
    """Return the compiled graph with the best available checkpointer."""
    global _compiled_graph
    if _compiled_graph is None:
        # Try MongoDB saver, fall back to in-memory, then no checkpointer
        checkpointer = None
        try:
            from ..checkpoints.mongodb_saver import get_checkpoint_saver

            checkpointer = get_checkpoint_saver()
        except Exception as e:
            logger.warning(f"MongoDB checkpointer unavailable: {e}")

        if checkpointer is None:
            try:
                from langgraph.checkpoint.memory import MemorySaver

                checkpointer = MemorySaver()
                logger.info("Using in-memory checkpointer")
            except Exception as e:
                logger.warning(f"MemorySaver unavailable: {e}")

        try:
            _compiled_graph = compile_graph(checkpointer=checkpointer)
        except Exception as e:
            logger.warning(f"Compiling without checkpointer: {e}")
            _compiled_graph = compile_graph()
    return _compiled_graph
