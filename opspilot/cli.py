"""OpsPilot CLI interface"""
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="OpsPilot - Task-aware AI agents")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_parser.add_argument('path', help='Path to documents')
    ingest_parser.add_argument('--recursive', '-r', action='store_true', 
                              help='Recursive directory traversal')
    ingest_parser.add_argument('--max-pdf-pages', type=int, default=10,
                              help='Maximum PDF pages (default: 10)')
    ingest_parser.add_argument('--max-log-size', type=int, default=100000,
                              help='Maximum log file size in bytes (default: 100KB)')
    ingest_parser.add_argument('--skip-oversized', action='store_true',
                              help='Skip files that exceed size limits instead of failing')
    
    # Agent command
    agent_parser = subparsers.add_parser('agent', help='Launch agent')
    agent_parser.add_argument('--task', required=True, help='Task description')
    agent_parser.add_argument('--context', help='Additional context')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'ingest':
        from opspilot.ingestion.ingestor import DocumentIngester
        
        try:
            ingester = DocumentIngester(
                max_pdf_pages=args.max_pdf_pages,
                max_log_size=args.max_log_size
            )
            
            if Path(args.path).is_file():
                doc = ingester.ingest_file(args.path)
                if doc:
                    print(f"✅ Ingested: {doc.metadata['title']}")
                    if doc.doc_type == 'pdf':
                        print(f"   📄 {doc.metadata['page_count']} pages")
                    elif doc.doc_type == 'log' and doc.metadata.get('was_truncated'):
                        print(f"   ⚠️  Log file was truncated (original: {doc.metadata['original_size']} bytes)")
            else:
                docs = ingester.ingest_directory(
                    args.path, 
                    recursive=args.recursive,
                    skip_oversized=args.skip_oversized
                )
                print(f"✅ Ingested {len(docs)} documents")
                
        except ValueError as e:
            print(f"❌ Error: {e}")
            if "pages" in str(e):
                print(f"💡 Use --max-pdf-pages to increase limit or --skip-oversized to skip large files")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            sys.exit(1)
    
    elif args.command == 'agent':
        print(f"🤖 Launching agent for task: {args.task}")
        if args.context:
            print(f"📝 Context: {args.context}")
        # Agent launch logic will go here

if __name__ == "__main__":
    main()
