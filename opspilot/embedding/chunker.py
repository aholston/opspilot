"""
OpsPilot Document Chunking and Embedding Module
Converts documents into vector embeddings for retrieval.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TextSplitter
)
from langchain_openai import OpenAIEmbeddings

from opspilot.ingestion.ingestor import Document


@dataclass
class Chunk:
    """Standardized chunk representation"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    doc_id: str
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Add derived metadata"""
        if 'chunk_size' not in self.metadata:
            self.metadata['chunk_size'] = len(self.content)
        if 'chunk_index' not in self.metadata:
            self.metadata['chunk_index'] = 0


class DocumentChunker(ABC):
    """Abstract base for document chunkers"""
    
    @abstractmethod
    def chunk_document(self, document: Document) -> List[Chunk]:
        pass


class SmartChunker(DocumentChunker):
    """Intelligent chunker that adapts to document type"""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 preserve_structure: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_structure = preserve_structure
        
        # Different splitters for different content types
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True
        )
        
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk document based on its type"""
        if document.doc_type == 'markdown' and self.preserve_structure:
            return self._chunk_markdown(document)
        elif document.doc_type == 'yaml':
            return self._chunk_yaml(document)
        elif document.doc_type == 'log':
            return self._chunk_log(document)
        else:
            return self._chunk_generic(document)
    
    def _chunk_markdown(self, document: Document) -> List[Chunk]:
        """Structure-aware markdown chunking"""
        chunks = []
        
        try:
            # First pass: split by headers
            md_header_splits = self.markdown_splitter.split_text(document.content)
            
            for i, split in enumerate(md_header_splits):
                # Second pass: split large sections
                if len(split.page_content) > self.chunk_size:
                    subsplits = self.text_splitter.split_text(split.page_content)
                    for j, subsplit in enumerate(subsplits):
                        chunk_metadata = {
                            **document.metadata,
                            **split.metadata,
                            'chunk_index': len(chunks),
                            'subsection': j if len(subsplits) > 1 else None,
                            'chunk_type': 'markdown_section'
                        }
                        
                        chunk_id = f"{document.metadata['doc_id']}_chunk_{len(chunks)}"
                        chunks.append(Chunk(
                            content=subsplit,
                            metadata=chunk_metadata,
                            chunk_id=chunk_id,
                            doc_id=document.metadata['doc_id']
                        ))
                else:
                    chunk_metadata = {
                        **document.metadata,
                        **split.metadata,
                        'chunk_index': len(chunks),
                        'chunk_type': 'markdown_section'
                    }
                    
                    chunk_id = f"{document.metadata['doc_id']}_chunk_{len(chunks)}"
                    chunks.append(Chunk(
                        content=split.page_content,
                        metadata=chunk_metadata,
                        chunk_id=chunk_id,
                        doc_id=document.metadata['doc_id']
                    ))
        
        except Exception:
            # Fallback to generic chunking
            return self._chunk_generic(document)
        
        return chunks
    
    def _chunk_yaml(self, document: Document) -> List[Chunk]:
        """YAML-aware chunking - often better as single chunk or by top-level keys"""
        content = document.content
        
        # If small enough, keep as single chunk
        if len(content) <= self.chunk_size:
            chunk_metadata = {
                **document.metadata,
                'chunk_index': 0,
                'chunk_type': 'yaml_complete'
            }
            
            chunk_id = f"{document.metadata['doc_id']}_chunk_0"
            return [Chunk(
                content=content,
                metadata=chunk_metadata,
                chunk_id=chunk_id,
                doc_id=document.metadata['doc_id']
            )]
        
        # Split by top-level sections if possible
        chunks = []
        lines = content.split('\n')
        current_section = []
        current_header = None
        
        for line in lines:
            # Detect top-level keys (no indentation)
            if line and not line.startswith(' ') and ':' in line:
                # Save previous section
                if current_section:
                    section_content = '\n'.join(current_section)
                    if len(section_content.strip()) > 0:
                        chunk_metadata = {
                            **document.metadata,
                            'chunk_index': len(chunks),
                            'yaml_section': current_header,
                            'chunk_type': 'yaml_section'
                        }
                        
                        chunk_id = f"{document.metadata['doc_id']}_chunk_{len(chunks)}"
                        chunks.append(Chunk(
                            content=section_content,
                            metadata=chunk_metadata,
                            chunk_id=chunk_id,
                            doc_id=document.metadata['doc_id']
                        ))
                
                # Start new section
                current_header = line.split(':')[0].strip()
                current_section = [line]
            else:
                current_section.append(line)
        
        # Add final section
        if current_section:
            section_content = '\n'.join(current_section)
            if len(section_content.strip()) > 0:
                chunk_metadata = {
                    **document.metadata,
                    'chunk_index': len(chunks),
                    'yaml_section': current_header,
                    'chunk_type': 'yaml_section'
                }
                
                chunk_id = f"{document.metadata['doc_id']}_chunk_{len(chunks)}"
                chunks.append(Chunk(
                    content=section_content,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                    doc_id=document.metadata['doc_id']
                ))
        
        return chunks if chunks else self._chunk_generic(document)
    
    def _chunk_log(self, document: Document) -> List[Chunk]:
        """Log-aware chunking - preserve log entry boundaries"""
        content = document.content
        chunks = []
        
        # Try to detect log entry patterns
        log_patterns = [
            r'^\d{4}-\d{2}-\d{2}',  # ISO date
            r'^\w{3}\s+\d{1,2}',    # Mon DD format
            r'^\[\d{4}-\d{2}-\d{2}',  # [YYYY-MM-DD]
            r'^\d{1,2}\/\d{1,2}\/\d{4}',  # MM/DD/YYYY
        ]
        
        lines = content.split('\n')
        current_chunk_lines = []
        
        for line in lines:
            # Check if line starts a new log entry
            is_new_entry = any(re.match(pattern, line) for pattern in log_patterns)
            
            if is_new_entry and current_chunk_lines:
                # Save current chunk if it's getting large
                current_content = '\n'.join(current_chunk_lines)
                if len(current_content) >= self.chunk_size:
                    chunk_metadata = {
                        **document.metadata,
                        'chunk_index': len(chunks),
                        'chunk_type': 'log_entries',
                        'entry_count': len([l for l in current_chunk_lines 
                                          if any(re.match(p, l) for p in log_patterns)])
                    }
                    
                    chunk_id = f"{document.metadata['doc_id']}_chunk_{len(chunks)}"
                    chunks.append(Chunk(
                        content=current_content,
                        metadata=chunk_metadata,
                        chunk_id=chunk_id,
                        doc_id=document.metadata['doc_id']
                    ))
                    current_chunk_lines = []
            
            current_chunk_lines.append(line)
        
        # Add final chunk
        if current_chunk_lines:
            current_content = '\n'.join(current_chunk_lines)
            chunk_metadata = {
                **document.metadata,
                'chunk_index': len(chunks),
                'chunk_type': 'log_entries',
                'entry_count': len([l for l in current_chunk_lines 
                                  if any(re.match(p, l) for p in log_patterns)])
            }
            
            chunk_id = f"{document.metadata['doc_id']}_chunk_{len(chunks)}"
            chunks.append(Chunk(
                content=current_content,
                metadata=chunk_metadata,
                chunk_id=chunk_id,
                doc_id=document.metadata['doc_id']
            ))
        
        return chunks if chunks else self._chunk_generic(document)
    
    def _chunk_generic(self, document: Document) -> List[Chunk]:
        """Generic text chunking fallback"""
        text_chunks = self.text_splitter.split_text(document.content)
        chunks = []
        
        for i, text_chunk in enumerate(text_chunks):
            chunk_metadata = {
                **document.metadata,
                'chunk_index': i,
                'chunk_type': 'generic'
            }
            
            chunk_id = f"{document.metadata['doc_id']}_chunk_{i}"
            chunks.append(Chunk(
                content=text_chunk,
                metadata=chunk_metadata,
                chunk_id=chunk_id,
                doc_id=document.metadata['doc_id']
            ))
        
        return chunks


class EmbeddingGenerator:
    """Handles embedding generation for chunks"""
    
    def __init__(self, 
                 provider: str = "openai",
                 model: str = "text-embedding-3-small",
                 api_key: Optional[str] = None):
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            self.embeddings = OpenAIEmbeddings(
                model=model,
                openai_api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}. Only 'openai' is currently supported.")
    
    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Generate embeddings for chunks"""
        texts = [chunk.content for chunk in chunks]
        
        try:
            embeddings = self.embeddings.embed_documents(texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
                chunk.metadata['embedding_model'] = self.model
                chunk.metadata['embedding_provider'] = self.provider
                
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise e
        
        return chunks
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query"""
        return self.embeddings.embed_query(query)


class DocumentProcessor:
    """Main processor combining chunking and embedding"""
    
    def __init__(self,
                 chunker: Optional[DocumentChunker] = None,
                 embedding_generator: Optional[EmbeddingGenerator] = None):
        self.chunker = chunker or SmartChunker()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
    
    def process_document(self, document: Document) -> List[Chunk]:
        """Process document into embedded chunks"""
        # Step 1: Chunk the document
        chunks = self.chunker.chunk_document(document)
        
        # Step 2: Generate embeddings
        embedded_chunks = self.embedding_generator.embed_chunks(chunks)
        
        print(f"📄 Processed {document.metadata['title']}: {len(embedded_chunks)} chunks")
        
        return embedded_chunks
    
    def process_documents(self, documents: List[Document]) -> List[Chunk]:
        """Process multiple documents"""
        all_chunks = []
        
        for document in documents:
            try:
                chunks = self.process_document(document)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"❌ Failed to process {document.metadata['title']}: {e}")
        
        print(f"✅ Total: {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


# Usage example
if __name__ == "__main__":
    from opspilot.ingestion.ingestor import DocumentIngester
    
    # Ingest documents
    ingester = DocumentIngester()
    documents = ingester.ingest_directory("./docs")
    
    # Process into chunks
    processor = DocumentProcessor()
    chunks = processor.process_documents(documents)
    
    # Print summary
    for chunk in chunks[:3]:  # Show first 3
        print(f"Chunk: {chunk.chunk_id}")
        print(f"Content: {chunk.content[:100]}...")
        print(f"Metadata: {chunk.metadata}")
        print("---")