"""Nodes components for the agentic RAG application."""

import logging
import time
from datetime import datetime
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from ..agents.tools import calculator, web_search
from ..core.llm import get_grader_llm, get_llm
from ..rag.crag import CRAGEvaluator
from ..rag.grader import AnswerRelevanceGrader, DocumentGrader, HallucinationGrader
from ..rag.self_rag import SelfRAGController
from ..retrieval.hybrid_search import get_hybrid_searcher
from .state import AgentState

logger = logging.getLogger(__name__)

# ─── Graders (shared instances) ──────────────────────────────────────────────
_doc_grader = DocumentGrader()
_hallucination_grader = HallucinationGrader()
_answer_grader = AnswerRelevanceGrader()
_crag = CRAGEvaluator()
_self_rag = SelfRAGController()


def _step(name: str, state: dict, extra: dict = None) -> dict:
    """Build a trace entry with a compact state snapshot."""
    return {
        "node": name,
        "timestamp": datetime.utcnow().isoformat(),
        "state_snapshot": {
            k: str(v)[:200]
            for k, v in state.items()
            if k not in ("steps", "documents", "graded_documents")
        },
        **(extra or {}),
    }


# ─── Node 1: Understand Query ─────────────────────────────────────────────────
async def understand_query(state: AgentState) -> dict[str, Any]:
    """Analyze the query intent and extract key concepts."""
    query = state["query"]
    start = time.time()
    llm = get_grader_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Analyze the query intent and extract key concepts.
Return JSON: {{"intent": "...", "key_concepts": [...], "query_type": "factual|analytical|creative|math"}}""",
            ),
            ("human", "Query: {query}"),
        ]
    )
    import json

    try:
        chain = prompt | llm
        result = await chain.ainvoke({"query": query})
        text = result.content.strip()
        s, e = text.find("{"), text.rfind("}") + 1
        data = json.loads(text[s:e]) if s >= 0 else {}
    except Exception:
        data = {"intent": query, "key_concepts": [query], "query_type": "factual"}

    elapsed = round(time.time() - start, 3)
    return {
        "metadata": {**state.get("metadata", {}), "query_analysis": data},
        "steps": [
            _step("understand_query", state, {"elapsed_ms": int(elapsed * 1000), "analysis": data})
        ],
    }


# ─── Node 2: Route ────────────────────────────────────────────────────────────
async def route_query(state: AgentState) -> dict[str, Any]:
    """Choose calculator, web search, retrieval, or direct generation."""
    query = state["query"].lower()
    analysis = state.get("metadata", {}).get("query_analysis", {})
    query_type = analysis.get("query_type", "factual")

    # Math/calculator
    math_keywords = {
        "calculate",
        "compute",
        "solve",
        "equation",
        "integral",
        "derivative",
        "+",
        "-",
        "*",
        "/",
        "^",
        "sqrt",
        "log",
        "sin",
        "cos",
    }
    if query_type == "math" or any(k in query for k in math_keywords):
        decision = "calculator"
    # Web search triggers
    elif any(
        k in query for k in {"latest", "recent", "news", "today", "current", "2024", "2025", "2026"}
    ):
        decision = "web_search"
    # General LLM (no retrieval needed)
    elif query_type == "creative" or any(
        k in query for k in {"write a", "create a", "compose", "poem", "story"}
    ):
        decision = "general"
    else:
        # Self-RAG: ask LLM if retrieval is needed
        should_ret, reason = await _self_rag.should_retrieve(state["query"])
        decision = "vectorstore" if should_ret else "general"

    return {
        "route_decision": decision,
        "should_retrieve": decision in ("vectorstore",),
        "steps": [_step("route", state, {"decision": decision})],
    }


# ─── Node 3: Retrieve (Hybrid Search) ────────────────────────────────────────
async def retrieve(state: AgentState) -> dict[str, Any]:
    """Fetch candidate documents through hybrid retrieval."""
    query = state.get("rewritten_query") or state["query"]
    searcher = get_hybrid_searcher()
    try:
        docs = await searcher.search(query)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        docs = []
    return {
        "documents": docs,
        "steps": [_step("retrieve", state, {"num_docs": len(docs), "query": query})],
    }


# ─── Node 4: Web Search ───────────────────────────────────────────────────────
async def web_search_node(state: AgentState) -> dict[str, Any]:
    """Convert web search results into source documents."""
    query = state.get("rewritten_query") or state["query"]
    raw = await web_search(query)
    docs = []
    for r in raw:
        content = r.get("content", "") if isinstance(r, dict) else str(r)
        url = r.get("url", "") if isinstance(r, dict) else ""
        title = r.get("title", "Web Result") if isinstance(r, dict) else "Web Result"
        if content:
            docs.append(
                Document(
                    page_content=content,
                    metadata={"source": url or "web", "title": title, "loader": "web_search"},
                )
            )
    return {
        "web_search_results": docs,
        "documents": docs,
        "steps": [_step("web_search", state, {"num_results": len(docs)})],
    }


# ─── Node 5: Calculator ───────────────────────────────────────────────────────
async def calculator_node(state: AgentState) -> dict[str, Any]:
    """Evaluate the mathematical expression extracted from the query."""
    query = state["query"]
    result = calculator.invoke(query)
    return {
        "calculator_result": result,
        "final_answer": result,
        "generation_quality": "good",
        "steps": [_step("calculator", state, {"result": result})],
    }


# ─── Node 6: Grade Documents (CRAG + Self-RAG) ───────────────────────────────
async def grade_documents(state: AgentState) -> dict[str, Any]:
    """Filter candidate documents and apply corrective retrieval."""
    query = state.get("rewritten_query") or state["query"]
    docs = state.get("documents", [])

    # CRAG evaluation
    crag_action, filtered_docs = await _crag.evaluate(query, docs)

    # Self-RAG: further filter
    if filtered_docs:
        self_rag_filtered = await _self_rag.filter_relevant_docs(query, filtered_docs)
        graded = self_rag_filtered if self_rag_filtered else filtered_docs
    else:
        graded = filtered_docs

    # If CRAG says web search needed, also do that
    if crag_action in ("web_search", "hybrid") and not state.get("web_search_results"):
        additional = await web_search(query)
        for r in additional:
            content = r.get("content", "") if isinstance(r, dict) else str(r)
            if content:
                graded.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": r.get("url", "web") if isinstance(r, dict) else "web",
                            "loader": "web_search",
                            "crag_added": True,
                        },
                    )
                )

    return {
        "graded_documents": graded,
        "crag_decision": crag_action,
        "steps": [
            _step(
                "grade_documents",
                state,
                {
                    "crag_action": crag_action,
                    "graded_count": len(graded),
                    "original_count": len(docs),
                },
            )
        ],
    }


# ─── Node 7: Generate ─────────────────────────────────────────────────────────
GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant. Answer the question using the provided context.
Be accurate, concise, and cite specific parts. Include source references at the end.

Context:
{context}

Instructions:
- Answer based only on the context provided
- If context is insufficient, say so clearly
- Include [Source: filename/URL] citations inline
- End with a "Sources:" section listing all referenced documents""",
        ),
        ("human", "{question}"),
    ]
)


async def generate(state: AgentState) -> dict[str, Any]:
    """Generate an answer from retrieved context or general knowledge."""
    question = state.get("rewritten_query") or state["query"]
    docs = state.get("graded_documents") or state.get("documents", [])
    llm = get_llm(temperature=0.3)

    if not docs and state.get("route_decision") == "general":
        # Direct LLM response
        from langchain_core.prompts import ChatPromptTemplate as CPT

        direct_prompt = CPT.from_messages(
            [
                ("system", "You are a helpful assistant. Answer clearly and helpfully."),
                ("human", "{question}"),
            ]
        )
        try:
            result = await (direct_prompt | llm).ainvoke({"question": question})
            generation = result.content
        except Exception as e:
            generation = f"I encountered an error: {e}"
        return {
            "generation": generation,
            "final_answer": generation,
            "sources": [],
            "steps": [_step("generate", state, {"mode": "direct_llm"})],
        }

    # Format context
    context_parts = []
    sources = []
    for i, doc in enumerate(docs[:6]):
        source = doc.metadata.get("source", f"doc_{i}")
        page = doc.metadata.get("page", "")
        ref = f"{source}" + (f" p.{page}" if page else "")
        context_parts.append(f"[Source {i + 1}: {ref}]\n{doc.page_content[:500]}")
        if source not in sources:
            sources.append(source)

    context = "\n\n---\n\n".join(context_parts)
    try:
        chain = GENERATION_PROMPT | llm
        result = await chain.ainvoke({"context": context, "question": question})
        generation = result.content
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        generation = f"I encountered an error generating the response: {e}"

    return {
        "generation": generation,
        "sources": sources,
        "steps": [_step("generate", state, {"mode": "rag", "num_sources": len(sources)})],
    }


# ─── Node 8: Evaluate / Grade Generation ─────────────────────────────────────
async def evaluate_generation(state: AgentState) -> dict[str, Any]:
    """Grade answer grounding, relevance, and usefulness."""
    generation = state.get("generation", "")
    query = state["query"]
    docs = state.get("graded_documents") or state.get("documents", [])

    if not generation:
        return {
            "generation_quality": "poor",
            "hallucination_score": 0.0,
            "answer_relevance_score": 0.0,
            "steps": [_step("evaluate", state, {"quality": "poor", "reason": "no generation"})],
        }

    # Check hallucination
    grounded, halluc_score, unsupported = await _hallucination_grader.grade(generation, docs)

    # Check answer relevance
    relevant, rel_score, rel_reason = await _answer_grader.grade(query, generation)

    # Self-RAG: IsSUP and IsUSE
    passages = [d.page_content for d in docs[:3]]
    _, sup_score, _ = await _self_rag.is_supported(passages, generation)
    utility, _ = await _self_rag.is_useful(query, generation)

    # Determine quality
    quality = "good" if (grounded and relevant and utility >= 3) else "poor"

    return {
        "generation_quality": quality,
        "hallucination_score": halluc_score,
        "answer_relevance_score": rel_score,
        "grounding_score": sup_score,
        "self_rag_issup": grounded,
        "self_rag_isuse": utility,
        "steps": [
            _step(
                "evaluate",
                state,
                {
                    "quality": quality,
                    "hallucination_score": halluc_score,
                    "answer_relevance_score": rel_score,
                    "utility": utility,
                    "unsupported": unsupported[:2] if unsupported else [],
                },
            )
        ],
    }


# ─── Node 9: Rewrite Query ────────────────────────────────────────────────────
async def rewrite_query(state: AgentState) -> dict[str, Any]:
    """Rewrite a query to improve retrieval on the next attempt."""
    original = state["query"]
    rewritten = state.get("rewritten_query", original)
    attempt = state.get("retry_count", 0)
    llm = get_grader_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Improve this search query for better document retrieval.
Return only the improved query, nothing else.
Attempt {attempt}: make it more specific, use different keywords, or broaden the scope.""",
            ),
            ("human", "Original query: {query}\nPrevious rewrite: {rewritten}"),
        ]
    )
    try:
        chain = prompt | llm
        result = await chain.ainvoke(
            {
                "query": original,
                "rewritten": rewritten,
                "attempt": attempt + 1,
            }
        )
        new_query = result.content.strip().strip('"').strip("'")
        if not new_query:
            new_query = original
    except Exception as e:
        logger.error(f"Query rewrite failed: {e}")
        new_query = original

    return {
        "rewritten_query": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "steps": [
            _step(
                "rewrite_query",
                state,
                {
                    "original": original,
                    "rewritten": new_query,
                    "retry_count": state.get("retry_count", 0) + 1,
                },
            )
        ],
    }


# ─── Node 10: Final Answer ────────────────────────────────────────────────────
async def final_answer(state: AgentState) -> dict[str, Any]:
    """Select the final response and record completion."""
    generation = state.get("generation", "")
    sources = state.get("sources", [])

    if not generation:
        if state.get("calculator_result"):
            return {
                "final_answer": state["calculator_result"],
                "steps": [_step("final_answer", state, {"type": "calculator"})],
            }
        generation = (
            "I was unable to generate a satisfactory answer. Please try rephrasing your question."
        )

    return {
        "final_answer": generation,
        "steps": [
            _step(
                "final_answer",
                state,
                {
                    "answer_length": len(generation),
                    "num_sources": len(sources),
                },
            )
        ],
    }
