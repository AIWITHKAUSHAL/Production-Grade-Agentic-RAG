"""Pipeline components for the agentic RAG application."""

import hashlib
import logging
from pathlib import Path
from typing import Any

from ..retrieval.bm25_retriever import BM25Index
from ..retrieval.vector_store import get_vector_store
from .chunking import get_chunker
from .document_loader import DocumentLoader

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """End-to-end document ingestion: load → chunk → embed → store."""

    def __init__(self):
        """Initialize the IngestionPipeline dependencies and state."""
        self.loader = DocumentLoader()
        self.vector_store = None
        self.bm25_index = None

    async def ingest_file(
        self,
        file_path: str,
        chunking_strategy: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        collection_name: str | None = None,
    ) -> dict[str, Any]:
        """Load, chunk, embed, and index a file for retrieval."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Ingesting {path.name} with strategy={chunking_strategy}")

        # Load document
        raw_docs = self.loader.load(str(path))
        if not raw_docs:
            raise ValueError(f"No content extracted from {path.name}")

        # Chunk documents
        chunker = get_chunker(chunking_strategy, chunk_size, chunk_overlap)
        chunks = chunker.chunk(raw_docs)

        # Enrich metadata
        file_hash = self._file_hash(path)
        for i, chunk in enumerate(chunks):
            chunk.metadata["file_hash"] = file_hash
            chunk.metadata["doc_id"] = f"{file_hash}_{i}"
            chunk.metadata["total_chunks"] = len(chunks)

        # Store in vector DB
        vs = await self._get_vector_store(collection_name)
        doc_ids = await vs.aadd_documents(chunks)

        # Update BM25 index
        bm25 = await self._get_bm25_index()
        await bm25.add_documents(chunks)

        logger.info(f"Ingested {len(chunks)} chunks from {path.name}")
        return {
            "file": path.name,
            "chunks": len(chunks),
            "pages": len(raw_docs),
            "strategy": chunking_strategy,
            "doc_ids": doc_ids[:5],  # Return first 5 IDs
            "file_hash": file_hash,
        }

    async def _get_vector_store(self, collection_name: str | None = None):
        """Initialize and return the pipeline vector store."""
        if self.vector_store is None:
            self.vector_store = await get_vector_store(collection_name)
        return self.vector_store

    async def _get_bm25_index(self) -> BM25Index:
        """Initialize and return the pipeline keyword index."""
        if self.bm25_index is None:
            self.bm25_index = BM25Index()
        return self.bm25_index

    def _file_hash(self, path: Path) -> str:
        """Compute a short SHA256 digest of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]


_pipeline: IngestionPipeline | None = None


def get_pipeline() -> IngestionPipeline:
    """Return the shared document ingestion pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline
