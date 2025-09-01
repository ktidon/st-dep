import streamlit as st
from langchain_community.vectorstores import Qdrant

@st.cache_resource
def create_vectorstore(_rag_documents, _embeddings, config):
    """Create vectorstore with fallback options"""
    print(f"DEBUG: create_vectorstore called with {len(_rag_documents) if _rag_documents else 0} documents")
    
    if not _rag_documents:
        st.error("No documents provided to vectorstore")
        return None
        
    if not _embeddings:
        st.error("No embeddings model provided to vectorstore")
        return None
    
    # Try multiple vectorstore options
    vectorstore_options = [
        ("Qdrant (in-memory)", create_qdrant_memory),
        ("FAISS", create_faiss_vectorstore),
        ("Chroma", create_chroma_vectorstore)
    ]
    
    for name, create_func in vectorstore_options:
        try:
            st.info(f"Trying {name}...")
            vectorstore = create_func(_rag_documents, _embeddings, config)
            if vectorstore:
                st.success(f"✅ {name} vectorstore created successfully!")
                return vectorstore
        except Exception as e:
            st.warning(f"❌ {name} failed: {e}")
            continue
    
    st.error("All vectorstore options failed!")
    return None

def create_qdrant_memory(_rag_documents, _embeddings, config):
    """Create Qdrant in-memory vectorstore"""
    from langchain_community.vectorstores import Qdrant
    
    # Test embeddings first
    test_embedding = _embeddings.embed_query("test")
    st.info(f"Embeddings working - dimension: {len(test_embedding)}")
    
    vectorstore = Qdrant.from_documents(
        documents=_rag_documents,
        embedding=_embeddings,
        location=":memory:",
        collection_name=config['rag']['collection_name']
    )
    
    # Test retrieval
    test_results = vectorstore.similarity_search("test query", k=1)
    st.info(f"Qdrant test: Found {len(test_results)} documents")
    
    return vectorstore

def create_faiss_vectorstore(_rag_documents, _embeddings, config):
    """Create FAISS vectorstore as backup"""
    from langchain_community.vectorstores import FAISS
    
    st.info("Creating FAISS vectorstore...")
    vectorstore = FAISS.from_documents(_rag_documents, _embeddings)
    
    # Test retrieval
    test_results = vectorstore.similarity_search("test query", k=1)
    st.info(f"FAISS test: Found {len(test_results)} documents")
    
    return vectorstore

def create_chroma_vectorstore(_rag_documents, _embeddings, config):
    """Create Chroma vectorstore as backup"""
    from langchain_community.vectorstores import Chroma
    import tempfile
    
    # Use temporary directory for Chroma
    temp_dir = tempfile.mkdtemp()
    st.info(f"Creating Chroma vectorstore in {temp_dir}")
    
    vectorstore = Chroma.from_documents(
        documents=_rag_documents,
        embedding=_embeddings,
        persist_directory=temp_dir
    )
    
    # Test retrieval
    test_results = vectorstore.similarity_search("test query", k=1)
    st.info(f"Chroma test: Found {len(test_results)} documents")
    
    return vectorstore

def add_documents_to_vectorstore(vectorstore, new_documents):
    """Add new documents to existing vectorstore"""
    try:
        vectorstore.add_documents(new_documents)
        return True
    except Exception as e:
        st.error(f"Error adding documents to vectorstore: {e}")
        return False