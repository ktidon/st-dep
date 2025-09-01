import streamlit as st
import pandas as pd
import os
from PIL import Image

# Local imports
from src.utils.config import load_config, check_api_key
from src.image_processing.classification import load_onnx_model, predict_image_class
from src.rag.document_loader import load_and_process_documents
from src.rag.vector_store import create_vectorstore
from src.rag.chains import create_rag_chain, create_vision_rag_chain
from src.rag.embeddings import initialize_llm_and_embeddings

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

# Load configuration
config = load_config()

# Main App Layout
st.title("🔬 MOSFET Die Crack Classification RAG System")
st.markdown("Upload an image for die crack classification and get AI-powered analysis with technical documentation.")

# Sidebar configuration
st.sidebar.header("🔑 Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=st.session_state.openai_api_key)
if openai_api_key:
    st.session_state.openai_api_key = openai_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key

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
    vectorstore = create_vectorstore(rag_documents, embeddings, config)

if not vectorstore:
    st.warning("⚠️ Vectorstore not available. Document-based queries will not work.")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 Image Upload")
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
                try:
                    response = technical_rag_chain.invoke(technical_question)
                    st.markdown("**Response:**")
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Error querying documentation: {e}")
    
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
                    try:
                        vision_input = {
                            "image_class": st.session_state.classification_result,
                            "confidence": st.session_state.confidence_score,
                            "question": vision_question
                        }
                        response = vision_rag_chain.invoke(vision_input)
                        st.markdown("**Analysis:**")
                        st.markdown(response)
                    except Exception as e:
                        st.error(f"Error analyzing classification: {e}")
        else:
            st.info("ℹ️ Classify an image first to enable image-specific analysis.")
else:
    st.warning("⚠️ Vectorstore not available. Please check your documents directory.")

# System Status
with st.expander("🔧 System Status"):
    st.write("**Component Status:**")
    status_data = {
        "Component": ["OpenAI API", "ONNX Model", "Vectorstore", "Documents"],
        "Status": [
            "✅ Connected" if st.session_state.openai_api_key else "❌ Not configured",
            "✅ Ready" if os.path.exists(model_path) else "❌ Model not found",
            "✅ Ready" if vectorstore else "❌ Not available",
            f"✅ {len(rag_documents)} docs loaded" if rag_documents else "❌ No documents"
        ]
    }
    st.table(pd.DataFrame(status_data))

# Footer
st.markdown("---")
st.markdown("*Powered by ONNX, LangChain, LangGraph, Qdrant, and OpenAI*")