import streamlit as st
import pandas as pd
import os
from PIL import Image
from datetime import datetime
import uuid

# Local imports
from src.utils.config import load_config, check_api_key
from src.image_processing.classification import load_onnx_model, predict_image_class
from src.rag.document_loader import load_and_process_documents
from src.rag.vector_store import create_vectorstore
from src.rag.chains import create_rag_chain, create_vision_rag_chain
from src.rag.embeddings import initialize_llm_and_embeddings

# LangSmith imports
from langsmith import Client
from langsmith.run_helpers import traceable

# Page configuration
st.set_page_config(
    page_title="MOSFET Die Crack Classification RAG System",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None
if 'confidence_score' not in st.session_state:
    st.session_state.confidence_score = None
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'langsmith_api_key' not in st.session_state:
    st.session_state.langsmith_api_key = ""
if 'langsmith_project' not in st.session_state:
    st.session_state.langsmith_project = f"MOSFET-RAG-{uuid.uuid4().hex[:8]}"
if 'langsmith_client' not in st.session_state:
    st.session_state.langsmith_client = None

# Load configuration
config = load_config()

# Main App Layout
st.title("🔬 MOSFET Die Crack Classification RAG System")
st.markdown("Upload an image for die crack classification and get AI-powered analysis with technical documentation.")

# Sidebar configuration
st.sidebar.header("🔑 Configuration")

# OpenAI API Key
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=st.session_state.openai_api_key)
if openai_api_key:
    st.session_state.openai_api_key = openai_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key

# LangSmith Configuration
st.sidebar.subheader("📊 LangSmith Tracking")
langsmith_api_key = st.sidebar.text_input("LangSmith API Key", type="password", value=st.session_state.langsmith_api_key)
if langsmith_api_key:
    st.session_state.langsmith_api_key = langsmith_api_key
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    
    # Initialize LangSmith client
    try:
        st.session_state.langsmith_client = Client()
        st.sidebar.success("✅ LangSmith tracking enabled")
    except Exception as e:
        st.sidebar.error(f"❌ LangSmith connection failed: {str(e)}")
        st.session_state.langsmith_client = None

langsmith_project = st.sidebar.text_input("LangSmith Project Name", value=st.session_state.langsmith_project)
if langsmith_project:
    st.session_state.langsmith_project = langsmith_project
    os.environ["LANGCHAIN_PROJECT"] = langsmith_project

# Other configuration
model_path = st.sidebar.text_input("ONNX Model Path", config['model']['path'])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, config['rag']['confidence_threshold'])
docs_path = st.sidebar.text_input("Documents Directory", config['documents']['static_path'])

# Check if API key is provided
if not check_api_key(st.session_state.openai_api_key):
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to proceed.")
    st.stop()

# Initialize components
generator_llm, embeddings = initialize_llm_and_embeddings()
if not generator_llm or not embeddings:
    st.error("Failed to initialize LLM or embeddings. Check your API key.")
    st.stop()

# Load documents and create vectorstore
with st.spinner("Loading documents and creating vectorstore..."):
    rag_documents = load_and_process_documents(docs_path)
    
    if rag_documents and embeddings:
        vectorstore = create_vectorstore(rag_documents, embeddings, config)
    else:
        st.error("Missing documents or embeddings for vectorstore creation")
        vectorstore = None

if not vectorstore:
    st.warning("⚠️ Vectorstore not available. Document-based queries will not work.")

# Helper function to log to LangSmith
def log_to_langsmith(query_type, input_data, output_data, metadata=None):
    """Log interactions to LangSmith"""
    if st.session_state.langsmith_client:
        try:
            # Create a run in LangSmith
            run_id = str(uuid.uuid4())
            
            # Prepare metadata
            run_metadata = {
                "query_type": query_type,
                "timestamp": datetime.now().isoformat(),
                "user_session": st.session_state.get('session_id', 'unknown'),
                **(metadata or {})
            }
            
            # Log the interaction
            st.session_state.langsmith_client.create_run(
                name=f"{query_type}_query",
                run_id=run_id,
                run_type="chain",
                inputs=input_data,
                outputs=output_data,
                project_name=st.session_state.langsmith_project,
                extra=run_metadata
            )
            
            return run_id
        except Exception as e:
            st.sidebar.warning(f"LangSmith logging failed: {str(e)}")
            return None
    return None

# Wrapped RAG functions with LangSmith tracking
@traceable(run_type="chain", name="technical_query")
def execute_technical_query(chain, question):
    """Execute technical query with LangSmith tracking"""
    try:
        response = chain.invoke(question)
        
        # Log to LangSmith
        log_to_langsmith(
            query_type="technical_documentation",
            input_data={"question": question},
            output_data={"answer": response},
            metadata={
                "vectorstore_used": vectorstore is not None,
                "confidence_threshold": confidence_threshold
            }
        )
        
        return response
    except Exception as e:
        st.error(f"Error during query: {str(e)}")
        return None

@traceable(run_type="chain", name="vision_query")
def execute_vision_query(chain, vision_input):
    """Execute vision query with LangSmith tracking"""
    try:
        response = chain.invoke(vision_input)
        
        # Log to LangSmith
        log_to_langsmith(
            query_type="image_specific_analysis",
            input_data={
                "image_class": vision_input.get("image_class"),
                "confidence": vision_input.get("confidence"),
                "question": vision_input.get("question")
            },
            output_data={"analysis": response},
            metadata={
                "classification": vision_input.get("image_class"),
                "confidence_score": vision_input.get("confidence")
            }
        )
        
        return response
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🖼️ Image Upload")
    uploaded_file = st.file_uploader(
        "Choose a die image for crack analysis",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        help="Upload a MOSFET die image for crack classification"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Die Image", use_container_width=True)
        
        # Classification button
        if st.button("🚀 Classify Die Crack", type="primary"):
            with st.spinner("Analyzing die image..."):
                try:
                    # Load model
                    session = load_onnx_model(model_path)
                    if session:
                        # Classify
                        predicted_class, confidence = predict_image_class(
                            session, image, config['model']['classes']
                        )
                        
                        if predicted_class and confidence:
                            # Store in session state
                            st.session_state.classification_result = predicted_class
                            st.session_state.confidence_score = confidence
                            
                            # Log classification to LangSmith
                            log_to_langsmith(
                                query_type="image_classification",
                                input_data={"image_name": uploaded_file.name},
                                output_data={
                                    "predicted_class": predicted_class,
                                    "confidence": confidence
                                },
                                metadata={
                                    "model_path": model_path,
                                    "image_size": image.size
                                }
                            )
                            
                            st.success(f"Classification complete!")
                        else:
                            st.error("Classification failed.")
                    
                except Exception as e:
                    st.error(f"Error during classification: {str(e)}")

with col2:
    st.header("📊 Classification Results")
    
    if st.session_state.classification_result:
        # Display classification results
        st.subheader("Die Crack Analysis")
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.metric("Classification", st.session_state.classification_result)
        with col2_2:
            st.metric("Confidence", f"{st.session_state.confidence_score:.2%}")
        
        # Confidence bar
        st.progress(st.session_state.confidence_score)
        
        # Confidence warning
        if st.session_state.confidence_score < confidence_threshold:
            st.warning(f"⚠️ Low confidence ({st.session_state.confidence_score:.2%}). Results may be unreliable.")

# RAG Query Section
st.header("🤖 Technical Analysis & Documentation Query")

# Create RAG chains
if vectorstore:
    technical_rag_chain = create_rag_chain(vectorstore, generator_llm)
    vision_rag_chain = create_vision_rag_chain(vectorstore, generator_llm)

    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.subheader("📚 General Technical Query")
        technical_question = st.text_area(
            "Ask about MOSFET die manufacturing, defects, or processes:",
            placeholder="What are the common causes of die cracks in MOSFET manufacturing?"
        )
        
        if st.button("🔍 Query Documentation") and technical_question:
            with st.spinner("Searching technical documentation..."):
                response = execute_technical_query(technical_rag_chain, technical_question)
                if response:
                    st.markdown("**Response:**")
                    st.markdown(response)
    
    with col4:
        st.subheader("🔬 Image-Specific Analysis")
        if st.session_state.classification_result:
            vision_question = st.text_area(
                "Ask about the classified image:",
                placeholder="What should I do about this die crack classification?",
                key="vision_question"
            )
            
            if st.button("🔍 Analyze Classification") and vision_question:
                with st.spinner("Analyzing classification with documentation..."):
                    vision_input = {
                        "image_class": st.session_state.classification_result,
                        "confidence": st.session_state.confidence_score,
                        "question": vision_question
                    }
                    response = execute_vision_query(vision_rag_chain, vision_input)
                    if response:
                        st.markdown("**Analysis:**")
                        st.markdown(response)
        else:
            st.info("ℹ️ Classify an image first to enable image-specific analysis.")
else:
    st.warning("⚠️ Vectorstore not available. Please check your documents directory.")

# System Status
with st.expander("🔧 System Status"):
    st.write("**Component Status:**")
    status_data = {
        "Component": ["OpenAI API", "LangSmith Tracking", "ONNX Model", "Vectorstore", "Documents"],
        "Status": [
            "✅ Connected" if st.session_state.openai_api_key else "❌ Not configured",
            "✅ Enabled" if st.session_state.langsmith_client else "❌ Not configured",
            "✅ Ready" if os.path.exists(model_path) else "❌ Model not found",
            "✅ Ready" if vectorstore else "❌ Not available",
            f"✅ {len(rag_documents)} docs loaded" if rag_documents else "❌ No documents"
        ]
    }
    st.table(pd.DataFrame(status_data))
    
    if st.session_state.langsmith_client:
        st.info(f"📊 LangSmith Project: **{st.session_state.langsmith_project}**")
        st.markdown(f"[View in LangSmith](https://smith.langchain.com)")

# Footer
st.markdown("---")
st.markdown("*Powered by ONNX, LangChain, LangGraph, Qdrant, OpenAI, and LangSmith*")