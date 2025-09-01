import streamlit as st
from langchain_community.vectorstores import Qdrant

@st.cache_resource
def create_vectorstore(_rag_documents, _embeddings, config):
    """Create Qdrant vectorstore"""
    print(f"DEBUG: create_vectorstore called with {len(_rag_documents) if _rag_documents else 0} documents")
    
    if not _rag_documents:
        st.error("No documents provided to vectorstore")
        return None
        
    if not _embeddings:
        st.error("No embeddings model provided to vectorstore")
        return None
    
    try:
        st.info(f"Creating vectorstore with {len(_rag_documents)} documents...")
        
        # Test embeddings first
        test_text = "test embedding"
        test_embedding = _embeddings.embed_query(test_text)
        st.success(f"Embeddings working - dimension: {len(test_embedding)}")
        
        # Create vectorstore
        vectorstore = Qdrant.from_documents(
            documents=_rag_documents,
            embedding=_embeddings,
            location=":memory:",
            collection_name=config['rag']['collection_name']
        )
        
        st.success(f"✅ Vectorstore created successfully with collection: {config['rag']['collection_name']}")
        
        # Test retrieval
        test_results = vectorstore.similarity_search("test query", k=1)
        st.info(f"Vectorstore test: Found {len(test_results)} similar documents")
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return None

def add_documents_to_vectorstore(vectorstore, new_documents):
    """Add new documents to existing vectorstore"""
    try:
        vectorstore.add_documents(new_documents)
        return True
    except Exception as e:
        st.error(f"Error adding documents to vectorstore: {e}")
        return False