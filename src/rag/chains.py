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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    def format_context(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    def create_retrieval_chain(input_dict):
        # Extract the question for retrieval
        question = input_dict.get("question", "")
        image_class = input_dict.get("image_class", "")
        
        # Create retrieval query combining question and classification
        retrieval_query = f"{question} {image_class}"
        docs = retriever.get_relevant_documents(retrieval_query)
        
        return {
            "context": format_context(docs),
            "image_class": image_class,
            "confidence": input_dict.get("confidence", 0.0),
            "question": question
        }
    
    vision_rag_chain = (
        create_retrieval_chain
        | vision_rag_prompt
        | ChatOpenAI(model="gpt-4o-mini", temperature=0)
        | StrOutputParser()
    )
    
    return vision_rag_chain