"""Tools components for the agentic RAG application."""

import logging
import math
import ssl

import certifi
import httpx
from langchain_core.tools import StructuredTool, tool

logger = logging.getLogger(__name__)


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use Python-safe math."""
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"


def get_web_search_tool(max_results: int = 3):
    """Create a Tavily search tool or a DuckDuckGo fallback."""
    from ..core.config import get_settings

    settings = get_settings()
    if settings.tavily_api_key:
        tls_context = ssl.create_default_context(cafile=certifi.where())
        headers = {"Authorization": f"Bearer {settings.tavily_api_key}"}

        def search(query: str) -> list[dict]:
            """Fetch Tavily search results synchronously with verified TLS."""
            with httpx.Client(verify=tls_context, timeout=30) as client:
                response = client.post(
                    "https://api.tavily.com/search",
                    headers=headers,
                    json={"query": query, "max_results": max_results, "search_depth": "basic"},
                )
                response.raise_for_status()
                return response.json()["results"]

        async def search_async(query: str) -> list[dict]:
            """Fetch Tavily search results asynchronously with verified TLS."""
            async with httpx.AsyncClient(verify=tls_context, timeout=30) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    headers=headers,
                    json={"query": query, "max_results": max_results, "search_depth": "basic"},
                )
                response.raise_for_status()
                return response.json()["results"]

        return StructuredTool.from_function(
            func=search,
            coroutine=search_async,
            name="tavily_search_results_json",
            description="Search the web for current information and return sources with content and URLs.",
        )
    # Fallback: DuckDuckGo (no API key needed)
    try:
        from langchain_community.tools import DuckDuckGoSearchRun

        return DuckDuckGoSearchRun()
    except Exception:
        return None


async def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Search the web and return content with source metadata."""
    try:
        search_tool = get_web_search_tool(max_results)
        if search_tool is None:
            return []
        results = await search_tool.ainvoke(query)
        if isinstance(results, str):
            return [{"content": results, "url": "", "title": "Web Search"}]
        if isinstance(results, list):
            return [
                result
                for result in results
                if isinstance(result, dict) and result.get("url") and result.get("content")
            ]
        return []
    except Exception as e:
        logger.error("Web search failed (%s)", type(e).__name__)
        return []


def get_all_tools():
    """Return the calculator and any available web search tool."""
    tools = [calculator]
    web = get_web_search_tool()
    if web:
        tools.append(web)
    return tools
