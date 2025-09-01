import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

@st.cache_resource
def load_and_process_documents(docs_path="documents/"):
    """Load and process documents for RAG"""
    if not os.path.exists(docs_path):
        st.warning(f"Documents directory '{docs_path}' not found. Using sample documents.")
        return []
    
    try:
        # Debug: Check what files are found
        import glob as glob_module
        pdf_files = glob_module.glob(os.path.join(docs_path, "**/*.pdf"), recursive=True)
        print(f"DEBUG: Found PDF files: {pdf_files}")
        
        if not pdf_files:
            st.warning(f"No PDF files found in {docs_path} or its subdirectories.")
            return []
        
        # Load documents
        loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
        documents = loader.load()
        
        st.info(f"Loaded {len(documents)} documents from {len(pdf_files)} PDF files.")
        
        if not documents:
            st.warning("No documents were successfully loaded.")
            return []
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=55
        )
        rag_documents = text_splitter.split_documents(documents)
        
        st.success(f"Created {len(rag_documents)} document chunks for RAG.")
        
        return rag_documents
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return []

def process_uploaded_documents(uploaded_files, upload_path="uploads/temp/"):
    """Process newly uploaded documents"""
    processed_docs = []
    
    for uploaded_file in uploaded_files:
        try:
            # Save to temp directory
            temp_path = os.path.join(upload_path, uploaded_file.name)
            os.makedirs(upload_path, exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and chunk the document
            loader = PyMuPDFLoader(temp_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=55
            )
            chunked_docs = text_splitter.split_documents(docs)
            processed_docs.extend(chunked_docs)
            
            # Move to processed directory
            processed_path = temp_path.replace("temp/", "processed/")
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            os.rename(temp_path, processed_path)
            
        except Exception as e:
            # Move to failed directory
            failed_path = temp_path.replace("temp/", "failed/")
            os.makedirs(os.path.dirname(failed_path), exist_ok=True)
            if os.path.exists(temp_path):
                os.rename(temp_path, failed_path)
            st.error(f"Error processing {uploaded_file.name}: {e}")
    
    return processed_docs