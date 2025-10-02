"""
One-time script to create and populate Qdrant collection with your documents.

Usage:
    python scripts/setup_qdrant_collection.py

Make sure to set environment variables:
    OPENAI_API_KEY=your-key
    QDRANT_URL=your-url
    QDRANT_API_KEY=your-key (if using cloud)
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

def setup_qdrant_collection(
    documents_path: str = "documents/",
    collection_name: str = "mosfet_docs",
    qdrant_url: str = None,
    qdrant_api_key: str = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """
    One-time setup: Load documents, embed them, and store in Qdrant.
    
    Args:
        documents_path: Path to your PDF documents
        collection_name: Name for the Qdrant collection
        qdrant_url: Qdrant server URL (default: from env)
        qdrant_api_key: Qdrant API key (default: from env)
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
    """
    
    # Get configuration
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set!")
    
    print("=" * 60)
    print("QDRANT COLLECTION SETUP")
    print("=" * 60)
    print(f"Documents path: {documents_path}")
    print(f"Collection name: {collection_name}")
    print(f"Qdrant URL: {qdrant_url}")
    print(f"Chunk size: {chunk_size}")
    print(f"Chunk overlap: {chunk_overlap}")
    print()
    
    # Step 1: Check Qdrant connection
    print("Step 1: Connecting to Qdrant...")
    try:
        if qdrant_api_key:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            client = QdrantClient(url=qdrant_url)
        
        # Test connection
        collections = client.get_collections()
        print(f"✅ Connected to Qdrant successfully!")
        print(f"   Existing collections: {[col.name for col in collections.collections]}")
        
        # Check if collection already exists
        if collection_name in [col.name for col in collections.collections]:
            response = input(f"\n⚠️  Collection '{collection_name}' already exists. Overwrite? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted.")
                return
            print(f"   Deleting existing collection '{collection_name}'...")
            client.delete_collection(collection_name)
            print(f"   ✅ Deleted.")
        
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        return
    
    # Step 2: Load documents
    print("\nStep 2: Loading documents...")
    try:
        if not os.path.exists(documents_path):
            raise ValueError(f"Documents path '{documents_path}' does not exist!")
        
        # Count PDF files
        import glob
        pdf_files = glob.glob(os.path.join(documents_path, "**/*.pdf"), recursive=True)
        print(f"   Found {len(pdf_files)} PDF files")
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in '{documents_path}'")
        
        # Load documents
        loader = DirectoryLoader(
            documents_path,
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} document pages")
        
    except Exception as e:
        print(f"❌ Failed to load documents: {e}")
        return
    
    # Step 3: Split documents into chunks
    print("\nStep 3: Splitting documents into chunks...")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} document chunks")
        
    except Exception as e:
        print(f"❌ Failed to split documents: {e}")
        return
    
    # Step 4: Initialize embeddings
    print("\nStep 4: Initializing OpenAI embeddings...")
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        # Test embeddings
        test_embedding = embeddings.embed_query("test")
        print(f"✅ Embeddings initialized (dimension: {len(test_embedding)})")
        
    except Exception as e:
        print(f"❌ Failed to initialize embeddings: {e}")
        return
    
    # Step 5: Create vectorstore and embed documents
    print("\nStep 5: Creating Qdrant collection and embedding documents...")
    print(f"⚠️  This will use OpenAI API credits to embed {len(chunks)} chunks!")
    
    # Estimate cost
    total_tokens = sum(len(chunk.page_content.split()) for chunk in chunks)
    estimated_cost = (total_tokens / 1000) * 0.0001  # $0.0001 per 1K tokens
    print(f"   Estimated tokens: ~{total_tokens}")
    print(f"   Estimated cost: ~${estimated_cost:.4f}")
    
    response = input("\nContinue? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return
    
    try:
        print("\n   Embedding documents (this may take a while)...")
        vectorstore = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
            force_recreate=True
        )
        
        # Verify collection
        collection_info = client.get_collection(collection_name)
        print(f"✅ Collection created successfully!")
        print(f"   Collection: {collection_name}")
        print(f"   Vectors: {collection_info.points_count}")
        print(f"   Dimension: {collection_info.config.params.vectors.size}")
        
    except Exception as e:
        print(f"❌ Failed to create vectorstore: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Test retrieval
    print("\nStep 6: Testing retrieval...")
    try:
        test_query = "What are MOSFET die cracks?"
        results = vectorstore.similarity_search(test_query, k=3)
        print(f"✅ Retrieval test successful!")
        print(f"   Query: '{test_query}'")
        print(f"   Found {len(results)} relevant documents")
        if results:
            print(f"   Top result preview: {results[0].page_content[:200]}...")
        
    except Exception as e:
        print(f"⚠️  Retrieval test failed: {e}")
    
    # Success!
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print(f"Collection '{collection_name}' is ready to use.")
    print(f"Total vectors: {collection_info.points_count}")
    print("\nYou can now run your Streamlit app:")
    print("  streamlit run app.py")
    print("\nThe app will connect to this collection without re-embedding.")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Qdrant collection with documents")
    parser.add_argument(
        "--documents",
        default="documents/",
        help="Path to documents directory (default: documents/)"
    )
    parser.add_argument(
        "--collection",
        default="mosfet_docs",
        help="Collection name (default: mosfet_docs)"
    )
    parser.add_argument(
        "--url",
        help="Qdrant URL (default: from QDRANT_URL env var)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Document chunk size (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Document chunk overlap (default: 200)"
    )
    
    args = parser.parse_args()
    
    setup_qdrant_collection(
        documents_path=args.documents,
        collection_name=args.collection,
        qdrant_url=args.url,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )