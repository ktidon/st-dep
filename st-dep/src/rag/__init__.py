"""
RAG (Retrieval-Augmented Generation) module for document processing and querying.

This module handles:
- Document loading and chunking
- Vector store operations with Qdrant
- LangChain RAG chains
- OpenAI embeddings and LLM integration
"""

from .document_loader import load_and_process_documents, process_uploaded_documents
from .vector_store import create_vectorstore, add_documents_to_vectorstore
from .chains import create_rag_chain, create_vision_rag_chain
from .embeddings import initialize_llm_and_embeddings

__all__ = [
    'load_and_process_documents',
    'process_uploaded_documents',
    'create_vectorstore',
    'add_documents_to_vectorstore',
    'create_rag_chain',
    'create_vision_rag_chain',
    'initialize_llm_and_embeddings'
]