import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

@st.cache_resource
def initialize_llm_and_embeddings():
    """Initialize OpenAI LLM and embeddings"""
    if not os.environ.get("OPENAI_API_KEY"):
        return None, None
    
    try:
        # Initialize OpenAI models
        generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
        generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
        embeddings = OpenAIEmbeddings()
        
        return generator_llm, embeddings
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None, None