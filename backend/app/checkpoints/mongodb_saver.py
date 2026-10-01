import logging
from typing import Any, Dict, Optional, Iterator
from datetime import datetime
from pymongo import MongoClient
from pymongo.collection import Collection

logger = logging.getLogger(__name__)


class MongoDBCheckpointSaver:
    """LangGraph checkpoint saver using MongoDB."""

    def __init__(self, uri: str, db_name: str, collection_name: str = "checkpoints"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection: Collection = self.db[collection_name]
        self._ensure_indexes()

    def _ensure_indexes(self):
        try:
            self.collection.create_index([("thread_id", 1), ("checkpoint_id", 1)], unique=True)
            self.collection.create_index("thread_id")
            self.collection.create_index("created_at")
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")

    def get(self, config: RunnableConfig) -> Optional[Dict[str, Any]]:
        thread_id = config["configurable"].get("thread_id")
        checkpoint_id = config["configurable"].get("checkpoint_id")
        if not thread_id:
            return None
        query = {"thread_id": thread_id}
        if checkpoint_id:
            query["checkpoint_id"] = checkpoint_id
        doc = self.collection.find_one(query, sort=[("created_at", -1)])
        if doc:
            doc.pop("_id", None)
            return doc
        return None

    def put(self, config: RunnableConfig, checkpoint: Dict[str, Any],
            metadata: Dict[str, Any] = None) -> RunnableConfig:
        thread_id = config["configurable"].get("thread_id")
        checkpoint_id = checkpoint.get("id", str(datetime.utcnow().timestamp()))
        doc = {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint": checkpoint,
            "metadata": metadata or {},
            "created_at": datetime.utcnow(),
        }
        self.collection.replace_one(
            {"thread_id": thread_id, "checkpoint_id": checkpoint_id},
            doc,
            upsert=True,
        )
        return {**config, "configurable": {**config.get("configurable", {}), "checkpoint_id": checkpoint_id}}

    def list(self, config: RunnableConfig) -> Iterator[Dict[str, Any]]:
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return
        for doc in self.collection.find({"thread_id": thread_id}, sort=[("created_at", -1)]):
            doc.pop("_id", None)
            yield doc

    def get_thread_history(self, thread_id: str) -> list:
        docs = list(self.collection.find({"thread_id": thread_id}, sort=[("created_at", 1)]))
        for d in docs:
            d.pop("_id", None)
        return docs

    def delete_thread(self, thread_id: str) -> int:
        result = self.collection.delete_many({"thread_id": thread_id})
        return result.deleted_count

    def close(self):
        self.client.close()


_checkpoint_saver: Optional[MongoDBCheckpointSaver] = None


def get_checkpoint_saver() -> Optional[MongoDBCheckpointSaver]:
    global _checkpoint_saver
    if _checkpoint_saver is None:
        try:
            from ..core.config import get_settings
            settings = get_settings()
            _checkpoint_saver = MongoDBCheckpointSaver(
                uri=settings.mongodb_uri,
                db_name=settings.mongodb_db,
                collection_name=settings.mongodb_checkpoint_collection,
            )
        except Exception as e:
            logger.error(f"MongoDB checkpoint saver unavailable: {e}")
    return _checkpoint_saver
