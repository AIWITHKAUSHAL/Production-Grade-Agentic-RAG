from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # EURI API
    euri_api_key: str = Field(..., env="EURI_API_KEY")
    euri_base_url: str = Field("https://api.euron.one/api/v1/euri", env="EURI_BASE_URL")
    euri_model: str = Field("gemini-3.5-flash-lite", env="EURI_MODEL")
    euri_embedding_model: str = Field("text-embedding-3-small", env="EURI_EMBEDDING_MODEL")

    # Qdrant Cloud
    qdrant_url: str = Field(..., env="QDRANT_URL")
    qdrant_api_key: str = Field(..., env="QDRANT_API_KEY")
    qdrant_collection: str = Field("agentic_rag_docs", env="QDRANT_COLLECTION")

    # MongoDB
    mongodb_uri: str = Field(..., env="MONGODB_URI")
    mongodb_db: str = Field("agentic_rag_db", env="MONGODB_DB")
    mongodb_checkpoint_collection: str = Field("checkpoints", env="MONGODB_CHECKPOINT_COLLECTION")

    # LangSmith
    langchain_tracing_v2: bool = Field(True, env="LANGCHAIN_TRACING_V2")
    langchain_api_key: Optional[str] = Field(None, env="LANGCHAIN_API_KEY")
    langchain_project: str = Field("production-agentic-rag", env="LANGCHAIN_PROJECT")
    langchain_endpoint: str = Field("https://api.smith.langchain.com", env="LANGCHAIN_ENDPOINT")

    # Web Search
    tavily_api_key: Optional[str] = Field(None, env="TAVILY_API_KEY")

    # Application
    app_host: str = Field("0.0.0.0", env="APP_HOST")
    app_port: int = Field(8000, env="APP_PORT")
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # Agent Settings
    max_retries: int = Field(3, env="MAX_RETRIES")
    max_documents: int = Field(10, env="MAX_DOCUMENTS")
    retrieval_k: int = Field(5, env="RETRIEVAL_K")
    rerank_top_k: int = Field(3, env="RERANK_TOP_K")
    hybrid_alpha: float = Field(0.5, env="HYBRID_ALPHA")  # 0=BM25, 1=semantic

    # Chunking
    default_chunk_size: int = Field(1000, env="DEFAULT_CHUNK_SIZE")
    default_chunk_overlap: int = Field(200, env="DEFAULT_CHUNK_OVERLAP")

    # Upload
    upload_dir: str = "uploads"
    max_upload_size_mb: int = 50

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
