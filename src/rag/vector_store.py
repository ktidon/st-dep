import streamlit as st
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os

@st.cache_resource
def create_vectorstore(_embeddings, config, qdrant_config=None):
    """Connect to existing Qdrant vectorstore (no re-embedding)"""
    
    if not _embeddings:
        st.error("No embeddings model provided")
        return None
    
    try:
        # Get Qdrant configuration
        qdrant_url = qdrant_config.get('url') if qdrant_config else os.environ.get("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = qdrant_config.get('api_key') if qdrant_config else os.environ.get("QDRANT_API_KEY")
        collection_name = config.get('rag', {}).get('collection_name', 'mosfet_docs')
        
        st.info(f"Connecting to Qdrant at {qdrant_url}, collection: {collection_name}")
        
        # Initialize Qdrant client
        if qdrant_api_key:
            client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=60
            )
        else:
            client = QdrantClient(
                url=qdrant_url,
                timeout=60
            )
        
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name not in collection_names:
            st.error(f"Collection '{collection_name}' not found in Qdrant!")
            st.info(f"Available collections: {collection_names}")
            return None
        
        # Get collection info
        collection_info = client.get_collection(collection_name)
        st.success(f"✅ Connected to collection '{collection_name}' with {collection_info.points_count} vectors")
        
        # Create vectorstore instance connected to existing collection
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=_embeddings
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
    
    try:
        # Get Qdrant configuration
        qdrant_url = qdrant_config.get('url') if qdrant_config else os.environ.get("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = qdrant_config.get('api_key') if qdrant_config else os.environ.get("QDRANT_API_KEY")
        collection_name = config.get('rag', {}).get('collection_name', 'mosfet_docs')
        
        st.warning(f"⚠️ Creating NEW collection and embedding {len(_rag_documents)} documents. This will use API credits!")
        
        # Initialize Qdrant client
        if qdrant_api_key:
            client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=60
            )
        else:
            client = QdrantClient(
                url=qdrant_url,
                timeout=60
            )
        
        # Create vectorstore with documents (this will embed them)
        vectorstore = QdrantVectorStore.from_documents(
            documents=_rag_documents,
            embedding=_embeddings,
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
            force_recreate=False  # Set to True to overwrite existing collection
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
        qdrant_url = qdrant_config.get('url') if qdrant_config else os.environ.get("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = qdrant_config.get('api_key') if qdrant_config else os.environ.get("QDRANT_API_KEY")
        
        if qdrant_api_key:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            client = QdrantClient(url=qdrant_url)
        
        collections = client.get_collections()
        return [col.name for col in collections.collections]
    except Exception as e:
        st.error(f"Error listing collections: {e}")
        return []