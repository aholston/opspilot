"""
OpsPilot Document Ingestion Module
Handles parsing and preprocessing of various document types.
"""

import os
import yaml
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

import PyPDF2
from markdown import markdown
from bs4 import BeautifulSoup


@dataclass
class Document:
    """Standardized document representation"""
    content: str
    metadata: Dict[str, Any]
    doc_type: str
    source_path: str
    
    def __post_init__(self):
        """Add default metadata"""
        if 'file_size' not in self.metadata:
            self.metadata['file_size'] = len(self.content)
        if 'doc_id' not in self.metadata:
            self.metadata['doc_id'] = self._generate_doc_id()
    
    def _generate_doc_id(self) -> str:
        """Generate unique document ID"""
        import hashlib
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        filename = Path(self.source_path).stem
        return f"{filename}_{content_hash}"


class DocumentProcessor(ABC):
    """Abstract base for document processors"""
    
    @abstractmethod
    def can_process(self, file_path: str) -> bool:
        pass
    
    @abstractmethod
    def process(self, file_path: str) -> Document:
        pass


class MarkdownProcessor(DocumentProcessor):
    def can_process(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.md', '.markdown'))
    
    def process(self, file_path: str) -> Document:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract frontmatter if present
        metadata = {}
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    metadata = yaml.safe_load(parts[1])
                    content = parts[2].strip()
                except yaml.YAMLError:
                    pass
        
        # Convert to plain text for better embedding
        html = markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        plain_text = soup.get_text()
        
        metadata.update({
            'title': self._extract_title(content),
            'headers': self._extract_headers(content)
        })
        
        return Document(
            content=plain_text,
            metadata=metadata,
            doc_type='markdown',
            source_path=file_path
        )
    
    def _extract_title(self, content: str) -> str:
        """Extract title from first h1 or filename"""
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith('# '):
                return line.strip()[2:]
        return Path(content).stem if hasattr(content, 'stem') else 'Untitled'
    
    def _extract_headers(self, content: str) -> List[str]:
        """Extract all headers for metadata"""
        headers = []
        lines = content.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                headers.append(stripped)
        return headers


class PDFProcessor(DocumentProcessor):
    def __init__(self, max_pages: int = 10):
        self.max_pages = max_pages
    
    def can_process(self, file_path: str) -> bool:
        return file_path.lower().endswith('.pdf')
    
    def process(self, file_path: str) -> Document:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            
            # Check page limit
            total_pages = len(reader.pages)
            if total_pages > self.max_pages:
                raise ValueError(
                    f"PDF has {total_pages} pages, exceeds limit of {self.max_pages}. "
                    f"Consider splitting the document or processing specific pages."
                )
            
            content = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    content.append(f"[Page {page_num + 1}]\n{text}")
        
        full_content = '\n\n'.join(content)
        
        metadata = {
            'page_count': total_pages,
            'title': Path(file_path).stem,
            'within_page_limit': True
        }
        
        return Document(
            content=full_content,
            metadata=metadata,
            doc_type='pdf',
            source_path=file_path
        )


class YAMLProcessor(DocumentProcessor):
    """Process YAML/config files - useful for infrastructure docs"""
    
    def can_process(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.yml', '.yaml'))
    
    def process(self, file_path: str) -> Document:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            parsed = yaml.safe_load(content)
            # Convert YAML to readable text format
            readable_content = self._yaml_to_text(parsed, Path(file_path).stem)
        except yaml.YAMLError:
            readable_content = f"Raw YAML content:\n{content}"
        
        metadata = {
            'title': Path(file_path).stem,
            'config_type': self._detect_config_type(file_path)
        }
        
        return Document(
            content=readable_content,
            metadata=metadata,
            doc_type='yaml',
            source_path=file_path
        )
    
    def _yaml_to_text(self, data: Any, filename: str) -> str:
        """Convert YAML data to searchable text"""
        def format_dict(d: dict, prefix: str = "") -> List[str]:
            lines = []
            for key, value in d.items():
                if isinstance(value, dict):
                    lines.append(f"{prefix}{key}:")
                    lines.extend(format_dict(value, prefix + "  "))
                elif isinstance(value, list):
                    lines.append(f"{prefix}{key}: {', '.join(map(str, value))}")
                else:
                    lines.append(f"{prefix}{key}: {value}")
            return lines
        
        if isinstance(data, dict):
            formatted_lines = [f"Configuration file: {filename}"] + format_dict(data)
            return '\n'.join(formatted_lines)
        else:
            return f"Configuration file: {filename}\nContent: {str(data)}"
    
    def _detect_config_type(self, file_path: str) -> str:
        """Detect what type of config this might be"""
        filename = Path(file_path).name.lower()
        if 'docker' in filename or 'compose' in filename:
            return 'docker'
        elif 'k8s' in filename or 'kubernetes' in filename:
            return 'kubernetes'
        elif 'ansible' in filename or 'playbook' in filename:
            return 'ansible'
        else:
            return 'generic'


class LogProcessor(DocumentProcessor):
    """Process log files with structure preservation"""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
    
    def can_process(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.log', '.txt'))
    
    def process(self, file_path: str) -> Document:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check size and sample if needed
        original_size = len(content)
        was_truncated = False
        
        if original_size > self.max_size:
            lines = content.split('\n')
            sampled_lines = (
                lines[:50] +  # First 50 lines
                [f'... [truncated: {original_size - self.max_size} chars omitted] ...'] +
                lines[-50:]   # Last 50 lines
            )
            content = '\n'.join(sampled_lines)
            was_truncated = True
        
        metadata = {
            'title': Path(file_path).stem,
            'log_type': self._detect_log_type(content),
            'line_count': len(content.split('\n')),
            'original_size': original_size,
            'was_truncated': was_truncated
        }
        
        return Document(
            content=content,
            metadata=metadata,
            doc_type='log',
            source_path=file_path
        )
    
    def _detect_log_type(self, content: str) -> str:
        """Basic log type detection"""
        content_lower = content.lower()
        if 'error' in content_lower or 'exception' in content_lower:
            return 'error_log'
        elif 'nginx' in content_lower or 'apache' in content_lower:
            return 'web_server'
        elif 'kubernetes' in content_lower or 'k8s' in content_lower:
            return 'kubernetes'
        else:
            return 'generic'


class DocumentIngester:
    """Main ingestion orchestrator"""
    
    def __init__(self, max_pdf_pages: int = 10, max_log_size: int = 100000):
        self.max_pdf_pages = max_pdf_pages
        self.max_log_size = max_log_size
        self.processors = [
            MarkdownProcessor(),
            PDFProcessor(max_pages=max_pdf_pages),
            YAMLProcessor(),
            LogProcessor(max_size=max_log_size)
        ]
    
    def ingest_file(self, file_path: str) -> Optional[Document]:
        """Ingest a single file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        for processor in self.processors:
            if processor.can_process(file_path):
                try:
                    return processor.process(file_path)
                except ValueError as e:
                    # Re-raise validation errors (like page limits)
                    raise e
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    return None
        
        print(f"No processor found for {file_path}")
        return None
    
    def ingest_directory(self, directory: str, recursive: bool = True, 
                        skip_oversized: bool = False) -> List[Document]:
        """Ingest all supported files in a directory"""
        documents = []
        skipped = []
        path = Path(directory)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        pattern = "**/*" if recursive else "*"
        for file_path in path.glob(pattern):
            if file_path.is_file():
                try:
                    doc = self.ingest_file(str(file_path))
                    if doc:
                        documents.append(doc)
                except ValueError as e:
                    if skip_oversized:
                        skipped.append((str(file_path), str(e)))
                        print(f"⚠️  Skipped {file_path}: {e}")
                    else:
                        raise e
        
        if skipped:
            print(f"\n📊 Summary: {len(documents)} documents ingested, {len(skipped)} skipped")
            
        return documents
    
    def add_processor(self, processor: DocumentProcessor):
        """Add custom document processor"""
        self.processors.append(processor)