import math
import logging
from typing import List
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

logger = logging.getLogger(__name__)


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use Python-safe math."""
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"


def get_web_search_tool(max_results: int = 3):
    from ..core.config import get_settings
    settings = get_settings()
    if settings.tavily_api_key:
        return TavilySearchResults(max_results=max_results, api_key=settings.tavily_api_key)
    # Fallback: DuckDuckGo (no API key needed)
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun()
    except Exception:
        return None


async def web_search(query: str, max_results: int = 3) -> List[dict]:
    from langchain_core.documents import Document
    tool = get_web_search_tool(max_results)
    if tool is None:
        return []
    try:
        results = await tool.ainvoke(query)
        if isinstance(results, str):
            return [{"content": results, "url": "", "title": "Web Search"}]
        if isinstance(results, list):
            return results
        return []
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []


def get_all_tools():
    tools = [calculator]
    web = get_web_search_tool()
    if web:
        tools.append(web)
    return tools
