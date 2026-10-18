"""Vector store components for the agentic RAG application."""

import logging

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import Distance, VectorParams

from ..core.config import get_settings
from ..core.embeddings import get_embeddings

logger = logging.getLogger(__name__)

_qdrant_client: QdrantClient | None = None
_async_qdrant_client: AsyncQdrantClient | None = None
_vector_store: QdrantVectorStore | None = None


def get_qdrant_client() -> QdrantClient:
    """Initialize and return the shared Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        settings = get_settings()
        _qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30,
        )
    return _qdrant_client


async def get_vector_store(collection_name: str | None = None) -> QdrantVectorStore:
    """Open the requested Qdrant collection with project embeddings."""
    global _vector_store
    settings = get_settings()
    coll = collection_name or settings.qdrant_collection
    embeddings = get_embeddings()

    client = get_qdrant_client()
    _ensure_collection(client, coll, vector_size=1536)

    _vector_store = QdrantVectorStore(
        client=client,
        collection_name=coll,
        embedding=embeddings,
        content_payload_key="content",
        metadata_payload_key="metadata",
    )
    return _vector_store


def _ensure_collection(client: QdrantClient, collection_name: str, vector_size: int = 1536):
    """Create a missing collection with the configured vector shape."""
    if client.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' already exists")
    else:
        logger.info(f"Creating collection '{collection_name}'")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def get_retriever(k: int = 5, collection_name: str | None = None):
    """Build a similarity retriever for the requested collection."""
    import asyncio

    loop = asyncio.new_event_loop()
    vs = loop.run_until_complete(get_vector_store(collection_name))
    loop.close()
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})


async def similarity_search(
    query: str, k: int = 5, filter_metadata: dict | None = None
) -> list[Document]:
    """Retrieve similar documents with an optional metadata filter."""
    vs = await get_vector_store()
    kwargs = {"k": k}
    if filter_metadata:
        kwargs["filter"] = filter_metadata
    return await vs.asimilarity_search(query, **kwargs)


async def similarity_search_with_score(query: str, k: int = 5) -> list[tuple[Document, float]]:
    """Retrieve similar documents with their vector scores."""
    vs = await get_vector_store()
    return await vs.asimilarity_search_with_score(query, k=k)
