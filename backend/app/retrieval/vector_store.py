import logging
from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant as LangchainQdrant
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from ..core.config import get_settings
from ..core.embeddings import get_embeddings

logger = logging.getLogger(__name__)

_qdrant_client: Optional[QdrantClient] = None
_async_qdrant_client: Optional[AsyncQdrantClient] = None
_vector_store: Optional[LangchainQdrant] = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        settings = get_settings()
        _qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30,
        )
    return _qdrant_client


async def get_vector_store(collection_name: Optional[str] = None) -> LangchainQdrant:
    global _vector_store
    settings = get_settings()
    coll = collection_name or settings.qdrant_collection
    embeddings = get_embeddings()

    client = get_qdrant_client()
    _ensure_collection(client, coll, vector_size=1536)

    _vector_store = LangchainQdrant(
        client=client,
        collection_name=coll,
        embeddings=embeddings,
        content_payload_key="content",
        metadata_payload_key="metadata",
    )
    return _vector_store


def _ensure_collection(client: QdrantClient, collection_name: str, vector_size: int = 1536):
    try:
        client.get_collection(collection_name)
        logger.info(f"Collection '{collection_name}' already exists")
    except Exception:
        logger.info(f"Creating collection '{collection_name}'")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def get_retriever(k: int = 5, collection_name: Optional[str] = None):
    import asyncio
    loop = asyncio.new_event_loop()
    vs = loop.run_until_complete(get_vector_store(collection_name))
    loop.close()
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})


async def similarity_search(query: str, k: int = 5,
                            filter_metadata: Optional[dict] = None) -> List[Document]:
    vs = await get_vector_store()
    kwargs = {"k": k}
    if filter_metadata:
        kwargs["filter"] = filter_metadata
    return await vs.asimilarity_search(query, **kwargs)


async def similarity_search_with_score(query: str, k: int = 5) -> List[tuple[Document, float]]:
    vs = await get_vector_store()
    return await vs.asimilarity_search_with_score(query, k=k)
