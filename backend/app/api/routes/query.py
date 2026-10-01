import uuid
import logging
import asyncio
import json
from typing import Optional, AsyncIterator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from ...agents.graph import get_graph
from ...agents.state import AgentState
from ...core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    stream: bool = False
    mode: str = "auto"  # auto | vectorstore | web_search | general


class QueryResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    sources: list[str] = []
    hallucination_score: float = 0.0
    answer_relevance_score: float = 0.0
    retry_count: int = 0
    route_decision: str = ""
    steps: list[dict] = []
    error: Optional[str] = None


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())
    graph = get_graph()
    settings = get_settings()

    initial_state: AgentState = {
        "query": request.query,
        "rewritten_query": None,
        "documents": [],
        "graded_documents": [],
        "web_search_results": [],
        "generation": None,
        "sources": [],
        "hallucination_score": 0.0,
        "answer_relevance_score": 0.0,
        "grounding_score": 0.0,
        "retry_count": 0,
        "route_decision": request.mode if request.mode != "auto" else "",
        "crag_decision": "",
        "should_retrieve": True,
        "generation_quality": "",
        "self_rag_isrel": False,
        "self_rag_issup": False,
        "self_rag_isuse": 0,
        "final_answer": None,
        "error": None,
        "calculator_result": None,
        "steps": [],
        "metadata": {},
        "session_id": session_id,
    }

    config = {"configurable": {"thread_id": session_id}}

    try:
        final_state = await graph.ainvoke(initial_state, config=config)
        return QueryResponse(
            session_id=session_id,
            query=request.query,
            answer=final_state.get("final_answer") or final_state.get("generation") or "No answer generated.",
            sources=final_state.get("sources", []),
            hallucination_score=final_state.get("hallucination_score", 0.0),
            answer_relevance_score=final_state.get("answer_relevance_score", 0.0),
            retry_count=final_state.get("retry_count", 0),
            route_decision=final_state.get("route_decision", ""),
            steps=final_state.get("steps", []),
            error=final_state.get("error"),
        )
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(500, f"Query processing failed: {str(e)}")


@router.post("/stream")
async def query_stream(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())
    graph = get_graph()

    initial_state: AgentState = {
        "query": request.query,
        "rewritten_query": None,
        "documents": [],
        "graded_documents": [],
        "web_search_results": [],
        "generation": None,
        "sources": [],
        "hallucination_score": 0.0,
        "answer_relevance_score": 0.0,
        "grounding_score": 0.0,
        "retry_count": 0,
        "route_decision": "",
        "crag_decision": "",
        "should_retrieve": True,
        "generation_quality": "",
        "self_rag_isrel": False,
        "self_rag_issup": False,
        "self_rag_isuse": 0,
        "final_answer": None,
        "error": None,
        "calculator_result": None,
        "steps": [],
        "metadata": {},
        "session_id": session_id,
    }

    config = {"configurable": {"thread_id": session_id}}

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event in graph.astream_events(initial_state, config=config, version="v1"):
                event_type = event.get("event", "")
                name = event.get("name", "")

                if event_type == "on_chain_start" and name not in ("LangGraph",):
                    yield f"data: {json.dumps({'type': 'node_start', 'node': name})}\n\n"

                elif event_type == "on_chain_end" and name not in ("LangGraph",):
                    output = event.get("data", {}).get("output", {})
                    steps = output.get("steps", []) if isinstance(output, dict) else []
                    yield f"data: {json.dumps({'type': 'node_end', 'node': name, 'steps': steps})}\n\n"

                elif event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk", {})
                    content = getattr(chunk, "content", "")
                    if content:
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/history/{session_id}")
async def get_session_history(session_id: str):
    try:
        from ...checkpoints.mongodb_saver import get_checkpoint_saver
        saver = get_checkpoint_saver()
        if saver:
            history = saver.get_thread_history(session_id)
            return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.warning(f"Could not fetch history: {e}")
    return {"session_id": session_id, "history": []}
