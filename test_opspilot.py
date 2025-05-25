#!/usr/bin/env python3
"""
OpsPilot Integration Test
Tests the complete pipeline from ingestion to agent interaction.
"""

import os
import sys
import shutil
from pathlib import Path

def create_test_docs():
    """Create sample documentation for testing"""
    test_dir = Path("test_docs")
    test_dir.mkdir(exist_ok=True)
    
    # Sample runbook
    with open(test_dir / "api_runbook.md", "w") as f:
        f.write("""# API Gateway Troubleshooting Runbook

## Common Issues

### Timeout Errors (504)
1. Check upstream service health
2. Verify connection pool settings
3. Review nginx configuration
4. Check database connectivity

### Authentication Failures
1. Verify JWT token validation
2. Check Redis session store
3. Review OAuth configuration
4. Validate SSL certificates

## Escalation
- P1: Page on-call SRE immediately
- P2: Create Slack incident channel
- Contact: sre-team@company.com
""")
    
    # Sample config
    with open(test_dir / "api-gateway.yaml", "w") as f:
        f.write("""# API Gateway Configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: api-gateway-config
data:
  nginx.conf: |
    upstream backend {
      server api-service:8080;
    }
    
    server {
      listen 80;
      location / {
        proxy_pass http://backend;
        proxy_timeout 30s;
      }
    }
  
  timeout_settings:
    connect_timeout: 5s
    read_timeout: 30s
    write_timeout: 30s
""")
    
    # Sample log
    with open(test_dir / "api.log", "w") as f:
        f.write("""2024-01-15 10:30:15 ERROR [api-gateway] Connection timeout to upstream server
2024-01-15 10:30:16 WARN [api-gateway] Retrying connection (attempt 2/3)
2024-01-15 10:30:17 ERROR [api-gateway] All retry attempts failed
2024-01-15 10:30:18 INFO [auth-service] JWT validation successful for user123
2024-01-15 10:30:19 ERROR [database] Connection pool exhausted
2024-01-15 10:30:20 ERROR [api-gateway] 504 Gateway Timeout returned to client
""")
    
    return test_dir

def test_imports():
    """Test that all imports work"""
    print("🔍 Testing imports...")
    
    try:
        from opspilot.ingestion.ingestor import DocumentIngester
        print("✅ Document ingestion import successful")
    except ImportError as e:
        print(f"❌ Ingestion import failed: {e}")
        return False
    
    try:
        from opspilot.embedding.chunker import DocumentProcessor
        print("✅ Embedding/chunking import successful")
    except ImportError as e:
        print(f"❌ Embedding import failed: {e}")
        return False
    
    try:
        from opspilot.storage.vector_store import create_vector_store
        print("✅ Vector store import successful")
    except ImportError as e:
        print(f"❌ Vector store import failed: {e}")
        return False
    
    try:
        from opspilot.tasks.parser import HybridTaskParser
        print("✅ Task parser import successful")
    except ImportError as e:
        print(f"❌ Task parser import failed: {e}")
        return False
    
    try:
        from opspilot.agents.launcher import AgentLauncher
        print("✅ Agent launcher import successful")
    except ImportError as e:
        print(f"❌ Agent launcher import failed: {e}")
        return False
    
    print("✅ All core imports successful")
    return True

def test_ingestion():
    """Test document ingestion"""
    print("\n📄 Testing document ingestion...")
    
    try:
        from opspilot.ingestion.ingestor import DocumentIngester
        
        test_dir = create_test_docs()
        ingester = DocumentIngester()
        docs = ingester.ingest_directory(str(test_dir))
        
        print(f"✅ Ingested {len(docs)} documents:")
        for doc in docs:
            print(f"   - {doc.metadata['title']} ({doc.doc_type})")
        
        return docs
        
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return []

def test_task_parsing():
    """Test task parsing without LLM"""
    print("\n🧠 Testing task parsing...")
    
    try:
        from opspilot.tasks.parser import HybridTaskParser
        
        parser = HybridTaskParser(use_llm=False)  # Rule-based only
        
        test_tasks = [
            "The API gateway is returning 504 errors",
            "Review the authentication service code changes",
            "Deploy the new version to staging"
        ]
        
        for task in test_tasks:
            result = parser.parse(task)
            print(f"✅ '{task}'")
            print(f"   Category: {result.category.value}")
            print(f"   Urgency: {result.urgency.value}")
            print(f"   Entities: {result.entities}")
        
        return True
        
    except Exception as e:
        print(f"❌ Task parsing failed: {e}")
        return False

def test_with_api_key():
    """Test complete pipeline with API key"""
    print("\n🤖 Testing complete pipeline (requires API key)...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  No OPENAI_API_KEY found - skipping LLM tests")
        return False
    
    try:
        from opspilot.ingestion.ingestor import DocumentIngester
        from opspilot.embedding.chunker import DocumentProcessor
        from opspilot.storage.vector_store import create_vector_store
        from opspilot.agents.launcher import AgentLauncher
        
        # Create test docs
        test_dir = create_test_docs()
        
        # Ingest documents
        print("  📄 Ingesting documents...")
        ingester = DocumentIngester()
        docs = ingester.ingest_directory(str(test_dir))
        
        # Process into chunks
        print("  ✂️  Creating chunks and embeddings...")
        processor = DocumentProcessor()
        chunks = processor.process_documents(docs)
        
        # Create vector store
        print("  🗄️  Building vector store...")
        vector_store = create_vector_store("faiss")
        vector_store.add_chunks(chunks)
        
        # Create agent launcher
        print("  🚀 Creating agent launcher...")
        launcher = AgentLauncher(vector_store=vector_store, api_key=api_key)
        
        # Test agent creation
        print("  🤖 Launching agent...")
        task = "The API gateway is returning 504 timeout errors and users can't login"
        agent = launcher.launch_agent(task)
        
        # Test interaction
        print("  💬 Testing agent response...")
        response = agent.respond("What should I check first?")
        
        print(f"✅ Agent responded with {len(response.content)} characters")
        print(f"   Sources used: {response.sources_used}")
        print(f"   Confidence: {response.confidence}")
        
        # Show first part of response
        preview = response.content[:200] + "..." if len(response.content) > 200 else response.content
        print(f"   Preview: {preview}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli():
    """Test CLI interface"""
    print("\n⌨️  Testing CLI...")
    
    try:
        from opspilot.cli import main
        print("✅ CLI module loads successfully")
        print("   To test CLI: python -m opspilot.cli ingest test_docs/")
        return True
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def cleanup():
    """Clean up test files"""
    print("\n🧹 Cleaning up...")
    
    # Remove test docs
    test_dir = Path("test_docs")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # Remove vector store if created
    vector_dir = Path("vector_store")
    if vector_dir.exists():
        shutil.rmtree(vector_dir)
    
    print("✅ Cleanup complete")

def main():
    """Run all tests"""
    print("🧪 OpsPilot Integration Test Suite")
    print("="*50)
    
    results = {}
    
    # Basic tests (no API key needed)
    results["imports"] = test_imports()
    if results["imports"]:
        docs = test_ingestion() 
        results["ingestion"] = len(docs) > 0 if docs else False
        results["task_parsing"] = test_task_parsing()
        results["cli"] = test_cli()
    
    # Advanced tests (API key needed)
    results["full_pipeline"] = test_with_api_key()
    
    # Summary
    print("\n📊 Test Results:")
    print("="*50)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:15} {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\nPassed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! OpsPilot is ready to use.")
    elif results.get("imports") and results.get("ingestion"):
        print("⚠️  Basic functionality works. Set OPENAI_API_KEY for full features.")
    else:
        print("❌ Some core functionality failed. Check error messages above.")
    
    # Cleanup
    cleanup()

if __name__ == "__main__":
    main()