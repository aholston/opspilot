"""OpsPilot - Task-aware AI agents for DevOps teams"""

__version__ = "0.1.0"

from opspilot.ingestion.ingestor import DocumentIngester, Document
from opspilot.storage.vector_store import VectorStore

__all__ = ["DocumentIngester", "Document", "VectorStore"]
