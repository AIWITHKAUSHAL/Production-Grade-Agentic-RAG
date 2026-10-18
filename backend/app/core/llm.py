"""Llm components for the agentic RAG application."""

from functools import lru_cache

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI

from .config import get_settings


def get_openai_client() -> OpenAI:
    """Create a synchronous client for the configured EURI API."""
    settings = get_settings()
    return OpenAI(
        api_key=settings.euri_api_key,
        base_url=settings.euri_base_url,
    )


def get_async_openai_client() -> AsyncOpenAI:
    """Create an asynchronous client for the configured EURI API."""
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.euri_api_key,
        base_url=settings.euri_base_url,
    )


@lru_cache
def get_llm(temperature: float = 0.0, streaming: bool = False) -> ChatOpenAI:
    """Return a cached chat model with the requested generation settings."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.euri_model,
        openai_api_key=settings.euri_api_key,
        openai_api_base=settings.euri_base_url,
        temperature=temperature,
        streaming=streaming,
        max_retries=3,
    )


def get_grader_llm() -> ChatOpenAI:
    """Create a deterministic chat model for grading."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.euri_model,
        openai_api_key=settings.euri_api_key,
        openai_api_base=settings.euri_base_url,
        temperature=0.0,
        max_retries=3,
    )


def get_streaming_llm() -> ChatOpenAI:
    """Create a chat model that streams response tokens."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.euri_model,
        openai_api_key=settings.euri_api_key,
        openai_api_base=settings.euri_base_url,
        temperature=0.3,
        streaming=True,
        max_retries=3,
    )
