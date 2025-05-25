"""
OpsPilot Vector Storage Module
Handles storage and retrieval of embedded document chunks.
"""

import os
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from pathlib import Path

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS, Weaviate
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document as LangchainDocument

from ..embedding.chunker import Chunk, EmbeddingGenerator


class VectorStore(ABC):
    """Abstract base for vector storage backends"""
    
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Tuple[Chunk, float]]:
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        pass
    
    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> None:
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector storage for local/fast retrieval"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator, 
                 dimension: int = 1536,
                 index_type: str = "IndexFlatIP"):  # Inner Product for cosine similarity
        self.embedding_generator = embedding_generator
        self.dimension = dimension
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "IndexIVFFlat":
            # For larger datasets - requires training
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 centroids
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Store chunks and metadata separately
        self.chunks: List[Chunk] = []
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.doc_id_to_indices: Dict[str, List[int]] = {}
        
        # LangChain integration
        self.langchain_store: Optional[FAISS] = None
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the vector store"""
        if not chunks:
            return
        
        # Ensure all chunks have embeddings
        chunks_to_embed = [c for c in chunks if c.embedding is None]
        if chunks_to_embed:
            self.embedding_generator.embed_chunks(chunks_to_embed)
        
        # Prepare vectors
        vectors = np.array([chunk.embedding for chunk in chunks]).astype('float32')
        
        # Normalize for cosine similarity (if using IP index)
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(vectors)
        
        # Train index if needed (for IVF indices)
        if not self.index.is_trained:
            self.index.train(vectors)
        
        # Add to FAISS
        start_idx = len(self.chunks)
        self.index.add(vectors)
        
        # Store chunks and build lookup tables
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.chunk_metadata.append(chunk.metadata)
            
            # Track document-to-chunk mapping
            doc_id = chunk.doc_id
            if doc_id not in self.doc_id_to_indices:
                self.doc_id_to_indices[doc_id] = []
            self.doc_id_to_indices[doc_id].append(start_idx + i)
        
        print(f"✅ Added {len(chunks)} chunks to FAISS store (total: {len(self.chunks)})")
    
    def search(self, query: str, k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks"""
        if len(self.chunks) == 0:
            return []
        
        # Generate query embedding
        query_vector = np.array([self.embedding_generator.embed_query(query)]).astype('float32')
        
        # Normalize query vector for cosine similarity
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector, min(k * 2, len(self.chunks)))  # Get extra for filtering
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue
                
            chunk = self.chunks[idx]
            
            # Apply metadata filtering
            if filter_metadata:
                if not self._matches_filter(chunk.metadata, filter_metadata):
                    continue
            
            results.append((chunk, float(score)))
            
            if len(results) >= k:
                break
        
        return results
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria"""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        return True
    
    def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove all chunks from a specific document"""
        if doc_id not in self.doc_id_to_indices:
            return
        
        indices_to_remove = sorted(self.doc_id_to_indices[doc_id], reverse=True)
        
        # Remove from tracking
        del self.doc_id_to_indices[doc_id]
        
        # Remove chunks and metadata (reverse order to maintain indices)
        for idx in indices_to_remove:
            del self.chunks[idx]
            del self.chunk_metadata[idx]
        
        # Rebuild FAISS index (unfortunately FAISS doesn't support deletion)
        self._rebuild_index()
        
        print(f"🗑️  Removed {len(indices_to_remove)} chunks for document {doc_id}")
    
    def _rebuild_index(self) -> None:
        """Rebuild FAISS index after deletions"""
        if not self.chunks:
            # Reset to empty index
            if self.index_type == "IndexFlatIP":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            return
        
        # Recreate index with remaining chunks
        vectors = np.array([chunk.embedding for chunk in self.chunks]).astype('float32')
        
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(vectors)
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            self.index.train(vectors)
        
        self.index.add(vectors)
        
        # Rebuild doc_id mapping
        self.doc_id_to_indices = {}
        for i, chunk in enumerate(self.chunks):
            doc_id = chunk.doc_id
            if doc_id not in self.doc_id_to_indices:
                self.doc_id_to_indices[doc_id] = []
            self.doc_id_to_indices[doc_id].append(i)
    
    def save(self, path: str) -> None:
        """Save vector store to disk"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        # Save chunks and metadata
        with open(save_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        
        with open(save_path / "metadata.json", "w") as f:
            json.dump({
                "chunk_metadata": self.chunk_metadata,
                "doc_id_to_indices": self.doc_id_to_indices,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "total_chunks": len(self.chunks)
            }, f, indent=2)
        
        print(f"💾 Saved FAISS store to {path}")
    
    def load(self, path: str) -> None:
        """Load vector store from disk"""
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store not found at {path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(load_path / "index.faiss"))
        
        # Load chunks
        with open(load_path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        
        # Load metadata
        with open(load_path / "metadata.json", "r") as f:
            metadata = json.load(f)
            self.chunk_metadata = metadata["chunk_metadata"]
            self.doc_id_to_indices = metadata["doc_id_to_indices"]
            self.dimension = metadata["dimension"]
            self.index_type = metadata["index_type"]
        
        print(f"📂 Loaded FAISS store from {path} ({len(self.chunks)} chunks)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        doc_counts = {doc_id: len(indices) for doc_id, indices in self.doc_id_to_indices.items()}
        
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(self.doc_id_to_indices),
            "index_type": self.index_type,
            "dimension": self.dimension,
            "chunks_per_document": doc_counts,
            "avg_chunks_per_doc": sum(doc_counts.values()) / len(doc_counts) if doc_counts else 0
        }


class WeaviateVectorStore(VectorStore):
    """Weaviate-based vector storage for production/cloud deployment"""
    
    def __init__(self, 
                 embedding_generator: EmbeddingGenerator,
                 url: str = "http://localhost:8080",
                 api_key: Optional[str] = None,
                 class_name: str = "OpsPilotChunk"):
        self.embedding_generator = embedding_generator
        self.url = url
        self.api_key = api_key
        self.class_name = class_name
        
        try:
            import weaviate
            
            # Initialize Weaviate client
            if api_key:
                auth_config = weaviate.AuthApiKey(api_key=api_key)
                self.client = weaviate.Client(url=url, auth_client_secret=auth_config)
            else:
                self.client = weaviate.Client(url=url)
            
            # Create schema if it doesn't exist
            self._ensure_schema()
            
            print(f"🌐 Connected to Weaviate at {url}")
            
        except ImportError:
            raise ImportError("weaviate-client is required for WeaviateVectorStore")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Weaviate: {e}")
    
    def _ensure_schema(self):
        """Ensure the Weaviate schema exists"""
        schema = {
            "class": self.class_name,
            "description": "OpsPilot document chunks",
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Chunk content"
                },
                {
                    "name": "doc_id", 
                    "dataType": ["string"],
                    "description": "Source document ID"
                },
                {
                    "name": "chunk_id",
                    "dataType": ["string"], 
                    "description": "Unique chunk ID"
                },
                {
                    "name": "doc_type",
                    "dataType": ["string"],
                    "description": "Document type"
                },
                {
                    "name": "title",
                    "dataType": ["string"],
                    "description": "Document title"
                },
                {
                    "name": "chunk_index",
                    "dataType": ["int"],
                    "description": "Chunk position in document"
                },
                {
                    "name": "metadata_json",
                    "dataType": ["text"],
                    "description": "Additional metadata as JSON"
                }
            ],
            "vectorizer": "none"  # We provide our own vectors
        }
        
        # Check if class exists
        existing_schema = self.client.schema.get()
        class_exists = any(cls["class"] == self.class_name for cls in existing_schema["classes"])
        
        if not class_exists:
            self.client.schema.create_class(schema)
            print(f"📋 Created Weaviate schema for class '{self.class_name}'")
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to Weaviate"""
        if not chunks:
            return
        
        # Ensure all chunks have embeddings
        chunks_to_embed = [c for c in chunks if c.embedding is None]
        if chunks_to_embed:
            self.embedding_generator.embed_chunks(chunks_to_embed)
        
        # Batch upload to Weaviate
        with self.client.batch as batch:
            batch.batch_size = 100
            
            for chunk in chunks:
                properties = {
                    "content": chunk.content,
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "doc_type": chunk.metadata.get("doc_type", "unknown"),
                    "title": chunk.metadata.get("title", ""),
                    "chunk_index": chunk.metadata.get("chunk_index", 0),
                    "metadata_json": json.dumps(chunk.metadata)
                }
                
                batch.add_data_object(
                    data_object=properties,
                    class_name=self.class_name,
                    vector=chunk.embedding
                )
        
        print(f"✅ Added {len(chunks)} chunks to Weaviate")
    
    def search(self, query: str, k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks in Weaviate"""
        query_vector = self.embedding_generator.embed_query(query)
        
        # Build search query
        search_query = (
            self.client.query
            .get(self.class_name, ["content", "doc_id", "chunk_id", "doc_type", "title", "chunk_index", "metadata_json"])
            .with_near_vector({"vector": query_vector})
            .with_limit(k)
            .with_additional(["distance"])
        )
        
        # Add metadata filtering if provided
        if filter_metadata:
            where_conditions = []
            for key, value in filter_metadata.items():
                if key in ["doc_id", "doc_type", "title"]:
                    where_conditions.append({
                        "path": [key],
                        "operator": "Equal",
                        "valueString": str(value)
                    })
                elif key == "chunk_index":
                    where_conditions.append({
                        "path": [key], 
                        "operator": "Equal",
                        "valueInt": int(value)
                    })
            
            if where_conditions:
                if len(where_conditions) == 1:
                    where_filter = where_conditions[0]
                else:
                    where_filter = {
                        "operator": "And",
                        "operands": where_conditions
                    }
                search_query = search_query.with_where(where_filter)
        
        # Execute search
        results = search_query.do()
        
        # Convert to chunks
        chunks_with_scores = []
        if "data" in results and "Get" in results["data"] and self.class_name in results["data"]["Get"]:
            for item in results["data"]["Get"][self.class_name]:
                # Reconstruct chunk
                metadata = json.loads(item["metadata_json"])
                chunk = Chunk(
                    content=item["content"],
                    metadata=metadata,
                    chunk_id=item["chunk_id"],
                    doc_id=item["doc_id"]
                )
                
                # Get similarity score (Weaviate returns distance, lower = more similar)
                distance = item["_additional"]["distance"]
                similarity = 1.0 - distance  # Convert to similarity
                
                chunks_with_scores.append((chunk, similarity))
        
        return chunks_with_scores
    
    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all chunks for a document"""
        where_filter = {
            "path": ["doc_id"],
            "operator": "Equal", 
            "valueString": doc_id
        }
        
        result = self.client.batch.delete_objects(
            class_name=self.class_name,
            where=where_filter
        )
        
        print(f"🗑️  Deleted chunks for document {doc_id}")
    
    def save(self, path: str) -> None:
        """Weaviate persists automatically - this is a no-op"""
        print("💾 Weaviate data persists automatically")
    
    def load(self, path: str) -> None:
        """Weaviate loads automatically - this is a no-op"""
        print("📂 Weaviate data loads automatically")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Weaviate storage statistics"""
        # Get total count
        result = self.client.query.aggregate(self.class_name).with_meta_count().do()
        total_chunks = result["data"]["Aggregate"][self.class_name][0]["meta"]["count"] if result["data"]["Aggregate"][self.class_name] else 0
        
        # Get unique document count (approximate)
        doc_result = self.client.query.aggregate(self.class_name).with_group_by(["doc_id"]).do()
        unique_docs = len(doc_result["data"]["Aggregate"][self.class_name]) if doc_result["data"]["Aggregate"][self.class_name] else 0
        
        return {
            "total_chunks": total_chunks,
            "total_documents": unique_docs,
            "backend": "weaviate",
            "url": self.url,
            "class_name": self.class_name
        }


def create_vector_store(backend: str = "faiss", **kwargs) -> VectorStore:
    """Factory function to create vector stores"""
    embedding_generator = kwargs.pop("embedding_generator", EmbeddingGenerator())
    
    if backend == "faiss":
        return FAISSVectorStore(embedding_generator, **kwargs)
    elif backend == "weaviate":
        return WeaviateVectorStore(embedding_generator, **kwargs)
    else:
        raise ValueError(f"Unsupported vector store backend: {backend}")


# Usage example
if __name__ == "__main__":
    from ..ingestion.ingestor import DocumentIngester
    from ..embedding.chunker import DocumentProcessor
    
    # Ingest and process documents
    ingester = DocumentIngester()
    documents = ingester.ingest_directory("./docs")
    
    processor = DocumentProcessor()
    chunks = processor.process_documents(documents)
    
    # Create vector store and add chunks
    vector_store = create_vector_store("faiss")
    vector_store.add_chunks(chunks)
    
    # Search
    results = vector_store.search("kubernetes deployment error", k=3)
    for chunk, score in results:
        print(f"Score: {score:.3f}")
        print(f"Content: {chunk.content[:200]}...")
        print("---")
    
    # Save for later
    vector_store.save("./vector_store")