import os
from openai import OpenAI, AsyncOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from functools import lru_cache
from .config import get_settings


def get_openai_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        api_key=settings.euri_api_key,
        base_url=settings.euri_base_url,
    )


def get_async_openai_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.euri_api_key,
        base_url=settings.euri_base_url,
    )


@lru_cache()
def get_llm(temperature: float = 0.0, streaming: bool = False) -> ChatOpenAI:
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
    settings = get_settings()
    return ChatOpenAI(
        model=settings.euri_model,
        openai_api_key=settings.euri_api_key,
        openai_api_base=settings.euri_base_url,
        temperature=0.0,
        max_retries=3,
    )


def get_streaming_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.euri_model,
        openai_api_key=settings.euri_api_key,
        openai_api_base=settings.euri_base_url,
        temperature=0.3,
        streaming=True,
        max_retries=3,
    )
