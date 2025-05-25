"""Basic ingestion tests"""
import pytest
from pathlib import Path
from opspilot.ingestion.ingestor import DocumentIngester, MarkdownProcessor

def test_markdown_processor():
    processor = MarkdownProcessor()
    assert processor.can_process("test.md") == True
    assert processor.can_process("test.txt") == False

def test_document_ingester():
    ingester = DocumentIngester()
    assert len(ingester.processors) > 0
