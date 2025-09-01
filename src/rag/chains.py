import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import StrOutputParser
from langchain_openai import ChatOpenAI
from operator import itemgetter
from src.prompts.technical_prompts import TECHNICAL_RAG_PROMPT
from src.prompts.vision_prompts import VISION_RAG_PROMPT

def create_rag_chain(vectorstore, llm):
    """Create RAG chain for technical questions"""
    if not vectorstore or not llm:
        return None
    
    technical_rag_prompt = ChatPromptTemplate.from_template(TECHNICAL_RAG_PROMPT)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    rag_chain = (
        RunnableParallel({
            "context": retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs])),
            "question": RunnablePassthrough()
        })
        | technical_rag_prompt
        | ChatOpenAI(model="gpt-4o-mini", temperature=0)
        | StrOutputParser()
    )
    
    return rag_chain

def create_vision_rag_chain(vectorstore, llm):
    """Create Vision RAG chain for image classification results"""
    if not vectorstore or not llm:
        return None
    
    vision_rag_prompt = ChatPromptTemplate.from_template(VISION_RAG_PROMPT)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    vision_rag_chain = (
        RunnableParallel({
            "context": retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs])),
            "image_class": itemgetter("image_class"),
            "confidence": itemgetter("confidence"),
            "question": itemgetter("question")
        })
        | vision_rag_prompt
        | ChatOpenAI(model="gpt-4o-mini", temperature=0)
        | StrOutputParser()
    )
    
    return vision_rag_chain