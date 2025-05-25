#!/usr/bin/env python3
"""
Check OpsPilot dependencies
"""

def check_dependency(package_name, import_name=None):
    """Check if a package is available"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} - Run: pip install {package_name}")
        return False

def main():
    print("🔍 Checking OpsPilot Dependencies")
    print("="*40)
    
    # Core dependencies
    deps = [
        ("langchain", "langchain"),
        ("langchain-openai", "langchain_openai"), 
        ("langchain-anthropic", "langchain_anthropic"),
        ("langchain-community", "langchain_community"),
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("faiss-cpu", "faiss"),
        ("PyPDF2", "PyPDF2"),
        ("markdown", "markdown"),
        ("beautifulsoup4", "bs4"),
        ("pyyaml", "yaml"),
        ("numpy", "numpy"),
    ]
    
    missing = []
    for package, import_name in deps:
        if not check_dependency(package, import_name):
            missing.append(package)
    
    print("\n" + "="*40)
    if missing:
        print(f"❌ Missing {len(missing)} dependencies")
        print("\nInstall missing packages:")
        print(f"pip install {' '.join(missing)}")
    else:
        print("✅ All dependencies available!")
    
    # Check optional
    print("\n🔧 Optional Dependencies:")
    optional_deps = [
        ("weaviate-client", "weaviate"),
        ("faiss-gpu", "faiss"),  # GPU version
    ]
    
    for package, import_name in optional_deps:
        check_dependency(package, import_name)

if __name__ == "__main__":
    main()