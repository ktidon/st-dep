import streamlit as st
from qdrant_client import QdrantClient
import os

# Try to import langchain_qdrant, fall back to langchain_community if not available
try:
    from langchain_qdrant import QdrantVectorStore
    QDRANT_IMPORT = "langchain_qdrant"
except ImportError:
    try:
        from langchain_community.vectorstores import Qdrant as QdrantVectorStore
        QDRANT_IMPORT = "langchain_community"
        st.warning("⚠️ Using legacy Qdrant import. Consider installing: pip install langchain-qdrant")
    except ImportError:
        st.error("❌ Cannot import Qdrant. Please install: pip install langchain-qdrant qdrant-client")
        QdrantVectorStore = None
        QDRANT_IMPORT = None

@st.cache_resource
def create_vectorstore(_embeddings, config, qdrant_config=None):
    """Connect to existing Qdrant vectorstore (no re-embedding)"""
    
    if not _embeddings:
        st.error("No embeddings model provided")
        return None
    
    if QdrantVectorStore is None:
        st.error("Qdrant not available. Install: pip install langchain-qdrant qdrant-client")
        return None
    
    try:
        # Get Qdrant configuration with your specific setup
        qdrant_url = qdrant_config.get('url') if qdrant_config else os.environ.get("QDRANT_URL")
        qdrant_api_key = qdrant_config.get('api_key') if qdrant_config else os.environ.get("QDRANT_API_KEY")
        collection_name = qdrant_config.get('collection_name') if qdrant_config else os.environ.get("QDRANT_COLLECTION", config.get('rag', {}).get('collection_name', 'FMEA_Handbook'))
        
        if not qdrant_url:
            st.error("QDRANT_URL not configured. Please add it to secrets.")
            return None
        
        if not qdrant_api_key:
            st.error("QDRANT_API_KEY not configured. Please add it to secrets.")
            return None
        
        st.info(f"Connecting to Qdrant at {qdrant_url}, collection: {collection_name}")
        
        # Initialize Qdrant client with HTTPS support (port 443)
        use_https = ':443' in qdrant_url or qdrant_url.startswith('https://')
        
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60,
            https=use_https
        )
        
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name not in collection_names:
            st.error(f"Collection '{collection_name}' not found in Qdrant!")
            st.info(f"Available collections: {collection_names}")
            st.warning("Please create the collection first using the setup script or Advanced section.")
            return None
        
        # Get collection info
        collection_info = client.get_collection(collection_name)
        st.success(f"✅ Connected to collection '{collection_name}' with {collection_info.points_count} vectors")
        
        # Create vectorstore instance connected to existing collection
        # Handle both import types
        if QDRANT_IMPORT == "langchain_qdrant":
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=_embeddings
            )
        else:  # langchain_community
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embeddings=_embeddings
            )
        
        # Test retrieval to verify connection
        test_results = vectorstore.similarity_search("test", k=1)
        st.success(f"✅ Vectorstore connected and working! Test retrieved {len(test_results)} documents")
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Error connecting to Qdrant: {str(e)}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return None


def create_new_vectorstore_with_documents(_rag_documents, _embeddings, config, qdrant_config=None):
    """
    Create NEW Qdrant vectorstore and embed documents.
    WARNING: This will use OpenAI API credits to embed documents!
    Only use this for initial setup or when adding new documents.
    """
    
    if not _rag_documents:
        st.error("No documents provided")
        return None
        
    if not _embeddings:
        st.error("No embeddings model provided")
        return None
    
    if QdrantVectorStore is None:
        st.error("Qdrant not available. Install: pip install langchain-qdrant qdrant-client")
        return None
    
    try:
        # Get Qdrant configuration
        qdrant_url = qdrant_config.get('url') if qdrant_config else os.environ.get("QDRANT_URL")
        qdrant_api_key = qdrant_config.get('api_key') if qdrant_config else os.environ.get("QDRANT_API_KEY")
        collection_name = qdrant_config.get('collection_name') if qdrant_config else os.environ.get("QDRANT_COLLECTION", config.get('rag', {}).get('collection_name', 'FMEA_Handbook'))
        
        st.warning(f"⚠️ Creating NEW collection and embedding {len(_rag_documents)} documents. This will use API credits!")
        
        # Initialize Qdrant client with HTTPS support
        use_https = ':443' in qdrant_url or qdrant_url.startswith('https://')
        
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60,
            https=use_https
        )
        
        # Create vectorstore with documents (this will embed them)
        if QDRANT_IMPORT == "langchain_qdrant":
            vectorstore = QdrantVectorStore.from_documents(
                documents=_rag_documents,
                embedding=_embeddings,
                url=qdrant_url,
                api_key=qdrant_api_key,
                collection_name=collection_name,
                force_recreate=False,
                https=use_https
            )
        else:  # langchain_community
            from langchain_community.vectorstores import Qdrant
            vectorstore = Qdrant.from_documents(
                documents=_rag_documents,
                embedding=_embeddings,
                url=qdrant_url,
                api_key=qdrant_api_key,
                collection_name=collection_name,
                force_recreate=False
            )
        
        st.success(f"✅ Created new collection '{collection_name}' with {len(_rag_documents)} document chunks")
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return None


def add_documents_to_vectorstore(vectorstore, new_documents):
    """
    Add new documents to existing vectorstore.
    WARNING: This will use OpenAI API credits to embed new documents!
    """
    try:
        st.warning(f"⚠️ Embedding and adding {len(new_documents)} new documents. This will use API credits!")
        vectorstore.add_documents(new_documents)
        st.success(f"✅ Added {len(new_documents)} documents to vectorstore")
        return True
    except Exception as e:
        st.error(f"Error adding documents: {str(e)}")
        return False


def list_qdrant_collections(qdrant_config=None):
    """Helper function to list all Qdrant collections"""
    try:
        qdrant_url = qdrant_config.get('url') if qdrant_config else os.environ.get("QDRANT_URL")
        qdrant_api_key = qdrant_config.get('api_key') if qdrant_config else os.environ.get("QDRANT_API_KEY")
        
        use_https = ':443' in qdrant_url or qdrant_url.startswith('https://')
        
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            https=use_https
        )
        
        collections = client.get_collections()
        return [col.name for col in collections.collections]
    except Exception as e:
        st.error(f"Error listing collections: {e}")
        return []