"""MongoDB persistence using the supported LangGraph checkpoint protocol."""

import logging

import certifi
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

logger = logging.getLogger(__name__)


class MongoDBCheckpointSaver(MongoDBSaver):
    """Persist LangGraph checkpoints and expose session history."""

    def __init__(self, uri: str, db_name: str, collection_name: str = "checkpoints"):
        """Connect to MongoDB with trusted TLS and initialize checkpoint indexes."""
        options = {"tlsCAFile": certifi.where()} if uri.startswith("mongodb+srv://") else {}
        client = MongoClient(uri, serverSelectionTimeoutMS=10000, connectTimeoutMS=10000, **options)
        try:
            client.admin.command("ping")
            super().__init__(client, db_name=db_name, checkpoint_collection_name=collection_name)
        except Exception:
            client.close()
            raise

    def get_thread_history(self, thread_id: str) -> list:
        """Return a session checkpoint history in chronological order."""
        history = []
        for item in self.list({"configurable": {"thread_id": thread_id}}):
            history.append(
                {
                    "thread_id": thread_id,
                    "checkpoint_id": item.config["configurable"]["checkpoint_id"],
                    "checkpoint": item.checkpoint,
                    "metadata": item.metadata,
                    "created_at": item.checkpoint.get("ts"),
                }
            )
        return list(reversed(history))

    def close(self):
        """Close the MongoDB client connection."""
        self.client.close()


_checkpoint_saver: MongoDBCheckpointSaver | None = None


def get_checkpoint_saver() -> MongoDBCheckpointSaver | None:
    """Initialize and return the shared MongoDB checkpoint saver."""
    global _checkpoint_saver
    if _checkpoint_saver is None:
        from ..core.config import get_settings

        settings = get_settings()
        try:
            _checkpoint_saver = MongoDBCheckpointSaver(
                uri=settings.mongodb_uri,
                db_name=settings.mongodb_db,
                collection_name=settings.mongodb_checkpoint_collection,
            )
        except Exception as exc:
            logger.error("MongoDB checkpoint saver unavailable (%s)", type(exc).__name__)
            raise
    return _checkpoint_saver


def close_checkpoint_saver():
    """Close and clear the shared checkpoint saver."""
    global _checkpoint_saver
    if _checkpoint_saver is not None:
        _checkpoint_saver.close()
        _checkpoint_saver = None
