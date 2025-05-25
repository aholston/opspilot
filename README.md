# OpsPilot

A framework for launching task-aware AI agents pre-loaded with system understanding and documentation context.

## Features

- **Document Ingestion**: Support for Markdown, PDFs, YAML configs, logs
- **Vector Storage**: FAISS and Weaviate backends
- **Task Intelligence**: Automatic task parsing and context construction
- **Agent Framework**: LangChain-based agents with retrieval capabilities

## Quick Start

```bash
# Install
pip install -e .

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Ingest documents
opspilot ingest ./docs --recursive

# Launch agent
opspilot agent --task "triage alert" --context "production monitoring"
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black opspilot/
```
