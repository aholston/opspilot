"""Basic OpsPilot usage example"""

from opspilot import DocumentIngester
import os

def main():
    # Initialize ingester
    ingester = DocumentIngester()
    
    # Create sample document
    os.makedirs("sample_docs", exist_ok=True)
    with open("sample_docs/readme.md", "w") as f:
        f.write("""# Sample Documentation

This is a sample document for OpsPilot testing.

## Features
- Document ingestion
- Vector storage
- AI agents
""")
    
    # Ingest documents
    docs = ingester.ingest_directory("sample_docs")
    print(f"Ingested {len(docs)} documents")
    
    for doc in docs:
        print(f"- {doc.metadata['title']} ({doc.doc_type})")

if __name__ == "__main__":
    main()
