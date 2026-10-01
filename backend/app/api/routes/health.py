from fastapi import APIRouter
from datetime import datetime
from ...core.config import get_settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    }


@router.get("/components")
async def components_health():
    settings = get_settings()
    results = {}

    # Qdrant
    try:
        from ...retrieval.vector_store import get_qdrant_client
        client = get_qdrant_client()
        client.get_collections()
        results["qdrant"] = "healthy"
    except Exception as e:
        results["qdrant"] = f"unhealthy: {e}"

    # MongoDB
    try:
        from ...checkpoints.mongodb_saver import get_checkpoint_saver
        saver = get_checkpoint_saver()
        if saver:
            saver.client.server_info()
            results["mongodb"] = "healthy"
        else:
            results["mongodb"] = "not configured"
    except Exception as e:
        results["mongodb"] = f"unhealthy: {e}"

    # LLM
    try:
        from ...core.llm import get_openai_client
        client = get_openai_client()
        results["llm"] = "configured"
    except Exception as e:
        results["llm"] = f"unhealthy: {e}"

    return {
        "status": "healthy" if all(v == "healthy" or v == "configured" or v == "not configured" for v in results.values()) else "degraded",
        "components": results,
        "timestamp": datetime.utcnow().isoformat(),
    }
