import streamlit as st
from langchain_community.vectorstores import Qdrant

@st.cache_resource
def create_vectorstore(_rag_documents, _embeddings, config):
    """Create Qdrant vectorstore"""
    if not _rag_documents or not _embeddings:
        return None
    
    try:
        vectorstore = Qdrant.from_documents(
            documents=_rag_documents,
            embedding=_embeddings,
            location=":memory:",
            collection_name=config['rag']['collection_name']
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        return None

def add_documents_to_vectorstore(vectorstore, new_documents):
    """Add new documents to existing vectorstore"""
    try:
        vectorstore.add_documents(new_documents)
        return True
    except Exception as e:
        st.error(f"Error adding documents to vectorstore: {e}")
        return False