from enum import Enum
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from .document_loader import RawDocument


class ChunkingStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"
    MARKDOWN = "markdown"
    PARAGRAPH = "paragraph"


class DocumentChunker:
    """Multiple chunking strategies for optimal retrieval."""

    def __init__(self, strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
                 chunk_size: int = 1000, chunk_overlap: int = 200):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, raw_docs: List[RawDocument]) -> List[Document]:
        chunks = []
        for raw_doc in raw_docs:
            doc_chunks = self._chunk_document(raw_doc)
            chunks.extend(doc_chunks)
        return chunks

    def _chunk_document(self, raw_doc: RawDocument) -> List[Document]:
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._fixed_size_chunk(raw_doc)
        elif self.strategy == ChunkingStrategy.RECURSIVE:
            return self._recursive_chunk(raw_doc)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunk(raw_doc)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return self._sentence_chunk(raw_doc)
        elif self.strategy == ChunkingStrategy.MARKDOWN:
            return self._markdown_chunk(raw_doc)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return self._paragraph_chunk(raw_doc)
        return self._recursive_chunk(raw_doc)

    def _make_doc(self, text: str, raw_doc: RawDocument, chunk_idx: int) -> Document:
        meta = dict(raw_doc.metadata)
        meta["chunk_index"] = chunk_idx
        meta["chunk_strategy"] = self.strategy.value
        meta["chunk_size_config"] = self.chunk_size
        return Document(page_content=text.strip(), metadata=meta)

    def _fixed_size_chunk(self, raw_doc: RawDocument) -> List[Document]:
        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n",
        )
        texts = splitter.split_text(raw_doc.content)
        return [self._make_doc(t, raw_doc, i) for i, t in enumerate(texts) if t.strip()]

    def _recursive_chunk(self, raw_doc: RawDocument) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        texts = splitter.split_text(raw_doc.content)
        return [self._make_doc(t, raw_doc, i) for i, t in enumerate(texts) if t.strip()]

    def _semantic_chunk(self, raw_doc: RawDocument) -> List[Document]:
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            from langchain_openai import OpenAIEmbeddings
            from ..core.config import get_settings
            settings = get_settings()
            embeddings = OpenAIEmbeddings(
                model=settings.euri_embedding_model,
                openai_api_key=settings.euri_api_key,
                openai_api_base=settings.euri_base_url,
            )
            splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
            texts = splitter.split_text(raw_doc.content)
            return [self._make_doc(t, raw_doc, i) for i, t in enumerate(texts) if t.strip()]
        except Exception:
            return self._recursive_chunk(raw_doc)

    def _sentence_chunk(self, raw_doc: RawDocument) -> List[Document]:
        try:
            splitter = SentenceTransformersTokenTextSplitter(
                chunk_overlap=20,
                model_name="all-MiniLM-L6-v2",
            )
            texts = splitter.split_text(raw_doc.content)
            return [self._make_doc(t, raw_doc, i) for i, t in enumerate(texts) if t.strip()]
        except Exception:
            return self._recursive_chunk(raw_doc)

    def _markdown_chunk(self, raw_doc: RawDocument) -> List[Document]:
        headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        try:
            md_docs = md_splitter.split_text(raw_doc.content)
            final_docs = char_splitter.split_documents(md_docs)
            result = []
            for i, d in enumerate(final_docs):
                meta = dict(raw_doc.metadata)
                meta.update(d.metadata)
                meta["chunk_index"] = i
                meta["chunk_strategy"] = "markdown"
                result.append(Document(page_content=d.page_content, metadata=meta))
            return result if result else self._recursive_chunk(raw_doc)
        except Exception:
            return self._recursive_chunk(raw_doc)

    def _paragraph_chunk(self, raw_doc: RawDocument) -> List[Document]:
        paragraphs = [p.strip() for p in raw_doc.content.split("\n\n") if p.strip()]
        # Merge small paragraphs
        merged, current = [], ""
        for para in paragraphs:
            if len(current) + len(para) < self.chunk_size:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    merged.append(current)
                current = para
        if current:
            merged.append(current)
        return [self._make_doc(t, raw_doc, i) for i, t in enumerate(merged) if t.strip()]


def get_chunker(strategy: str = "recursive", chunk_size: int = 1000,
                chunk_overlap: int = 200) -> DocumentChunker:
    strategy_enum = ChunkingStrategy(strategy) if strategy in ChunkingStrategy._value2member_map_ else ChunkingStrategy.RECURSIVE
    return DocumentChunker(strategy=strategy_enum, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
