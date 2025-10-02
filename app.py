import streamlit as st
import pandas as pd
import os
from PIL import Image

# Local imports
from src.utils.config import load_config, check_api_key
from src.image_processing.classification import load_onnx_model, predict_image_class
from src.rag.document_loader import load_and_process_documents
from src.rag.vector_store import (
    create_vectorstore, 
    create_new_vectorstore_with_documents,
    add_documents_to_vectorstore,
    list_qdrant_collections
)
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
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# Helper function to get secrets
def get_secret(key, default=""):
    """Get secret from Streamlit secrets or environment variables"""
    try:
        return st.secrets.get(key, os.environ.get(key, default))
    except:
        return os.environ.get(key, default)

# Load configuration
config = load_config()

# Main App Layout
st.title("🔬 MOSFET Die Crack Classification RAG System")
st.markdown("Upload an image for die crack classification and get AI-powered analysis with technical documentation.")

# Sidebar configuration
st.sidebar.header("🔧 Configuration")

# API Key handling with secrets support
default_api_key = get_secret("OPENAI_API_KEY", "")
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key", 
    type="password", 
    value=default_api_key,
    help="Enter your OpenAI API key or configure in Streamlit secrets"
)

if openai_api_key:
    st.session_state.openai_api_key = openai_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Qdrant configuration with your specific setup
st.sidebar.subheader("🗄️ Qdrant Configuration")

# Default to your Qdrant Cloud URL
default_qdrant_url = get_secret("QDRANT_URL", "https://a56715f1-e0ff-43cc-819a-66f55e1c3a52.us-east-1.aws.cloud.qdrant.io:443")
qdrant_url = st.sidebar.text_input(
    "Qdrant URL", 
    value=default_qdrant_url,
    help="Your Qdrant Cloud URL (port 443 for HTTPS)"
)

default_qdrant_api_key = get_secret("QDRANT_API_KEY", "")
qdrant_api_key = st.sidebar.text_input(
    "Qdrant API Key", 
    type="password",
    value=default_qdrant_api_key,
    help="Your Qdrant Cloud API key"
)

# Default to your collection name - with proper fallback
default_collection = get_secret("QDRANT_COLLECTION", "")
if not default_collection:
    default_collection = "NexPert"  # Use one of your existing collections

collection_name = st.sidebar.text_input(
    "Collection Name",
    value=default_collection,
    help="Name of your Qdrant collection (Available: FMEA_Synth, NexPert)"
)

# Store Qdrant config in environment
if qdrant_url:
    os.environ["QDRANT_URL"] = qdrant_url
if qdrant_api_key:
    os.environ["QDRANT_API_KEY"] = qdrant_api_key
if collection_name:
    os.environ["QDRANT_COLLECTION"] = collection_name
    config['rag']['collection_name'] = collection_name

# Embedding model configuration
st.sidebar.subheader("🧠 Embedding Model")
embedding_model_option = st.sidebar.selectbox(
    "Embedding Model",
    options=[
        "text-embedding-3-large (3072-dim)",
        "text-embedding-ada-002 (1536-dim)"
    ],
    index=0,  # Default to 3-large to match FMEA_Synth
    help="Must match your collection's dimensions"
)

# Extract model name
if "3-large" in embedding_model_option:
    embedding_model = "text-embedding-3-large"
    expected_dim = 3072
else:
    embedding_model = "text-embedding-ada-002"
    expected_dim = 1536

st.sidebar.caption(f"Expected dimension: {expected_dim}")

# Model configuration
model_path = st.sidebar.text_input("ONNX Model Path", config['model']['path'])
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    0.0, 1.0, 
    config['rag']['confidence_threshold']
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Vectorstore Actions")

# List available collections
if st.sidebar.button("📋 List Collections"):
    with st.spinner("Fetching collections..."):
        qdrant_config = {
            'url': qdrant_url, 
            'api_key': qdrant_api_key if qdrant_api_key else None,
            'collection_name': collection_name
        }
        collections = list_qdrant_collections(qdrant_config)
        if collections:
            st.sidebar.success(f"Available collections: {', '.join(collections)}")
            if collection_name in collections:
                st.sidebar.info(f"✅ Your collection '{collection_name}' exists!")
            else:
                st.sidebar.warning(f"⚠️ Collection '{collection_name}' not found. Available: {', '.join(collections)}")
        else:
            st.sidebar.warning("No collections found or connection failed")

# Reconnect to vectorstore
if st.sidebar.button("🔌 Reconnect to Vectorstore"):
    st.session_state.vectorstore = None
    st.rerun()

# Advanced: Create new collection (with warning)
with st.sidebar.expander("⚠️ Advanced: Create New Collection"):
    st.warning("This will use API credits to embed documents!")
    docs_path = st.text_input("Documents Directory", config['documents']['static_path'])
    if st.button("🆕 Create New Collection", type="secondary"):
        with st.spinner("Loading and embedding documents..."):
            rag_documents = load_and_process_documents(docs_path)
            if rag_documents:
                qdrant_config = {
                    'url': qdrant_url, 
                    'api_key': qdrant_api_key if qdrant_api_key else None,
                    'collection_name': collection_name
                }
                generator_llm, embeddings = initialize_llm_and_embeddings()
                if embeddings:
                    vectorstore = create_new_vectorstore_with_documents(
                        rag_documents, embeddings, config, qdrant_config
                    )
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.rerun()

# Check if API key is provided
if not check_api_key(st.session_state.openai_api_key):
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to proceed.")
    st.stop()

# Initialize components (cached with embedding model as key)
@st.cache_resource
def get_llm_and_embeddings(_embedding_model):
    """Initialize LLM and embeddings (cached per model)"""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    
    try:
        st.info(f"Initializing embeddings with model: {_embedding_model}")
        generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
        embeddings = OpenAIEmbeddings(model=_embedding_model)
        
        # Test embeddings to verify dimension
        test_vec = embeddings.embed_query("test")
        st.success(f"✅ Embeddings loaded: {len(test_vec)} dimensions")
        
        return generator_llm, embeddings
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None, None

# Clear cache button in sidebar
if st.sidebar.button("🗑️ Clear Cache & Reload"):
    st.cache_resource.clear()
    st.rerun()

generator_llm, embeddings = get_llm_and_embeddings(embedding_model)
if not generator_llm or not embeddings:
    st.error("Failed to initialize LLM or embeddings. Check your API key.")
    st.stop()

# Connect to existing vectorstore (NO re-embedding)
if st.session_state.vectorstore is None:
    with st.spinner("Connecting to Qdrant vectorstore..."):
        try:
            qdrant_config = {
                'url': qdrant_url,
                'api_key': qdrant_api_key if qdrant_api_key else None,
                'collection_name': collection_name
            }
            
            # Connect to EXISTING vectorstore (no embedding)
            vectorstore = create_vectorstore(embeddings, config, qdrant_config)
            
            if vectorstore:
                st.session_state.vectorstore = vectorstore
            else:
                st.error("Failed to connect to vectorstore. Please check your Qdrant configuration.")
                
        except Exception as e:
            st.error(f"Error connecting to vectorstore: {str(e)}")
            st.session_state.vectorstore = None

vectorstore = st.session_state.vectorstore

if not vectorstore:
    st.warning("⚠️ Vectorstore not connected. Please check your Qdrant configuration in the sidebar.")
    st.info(f"💡 Tip: Make sure collection '{collection_name}' exists in your Qdrant Cloud instance.")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Image Upload")
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
                    # Load model (cached)
                    @st.cache_resource
                    def get_model(path):
                        return load_onnx_model(path)
                    
                    session = get_model(model_path)
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
                            st.rerun()
                        else:
                            st.error("Classification failed.")
                    else:
                        st.error("Failed to load ONNX model.")
                    
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
    else:
        st.info("ℹ️ Upload and classify an image to see results here.")

# RAG Query Section
st.header("🤖 Technical Analysis & Documentation Query")

# Create RAG chains
if vectorstore:
    try:
        technical_rag_chain = create_rag_chain(vectorstore, generator_llm)
        vision_rag_chain = create_vision_rag_chain(vectorstore, generator_llm)

        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.subheader("📚 General Technical Query")
            technical_question = st.text_area(
                "Ask about MOSFET die manufacturing, defects, or processes:",
                placeholder="What are the common causes of die cracks in MOSFET manufacturing?",
                key="tech_query"
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
    except Exception as e:
        st.error(f"Error creating RAG chains: {e}")
else:
    st.warning("⚠️ Vectorstore not connected. Please check your Qdrant configuration in the sidebar.")

# System Status
with st.expander("🔧 System Status"):
    st.write("**Component Status:**")
    
    # Check Qdrant connection
    qdrant_status = "❌ Not connected"
    points_count = 0
    if vectorstore:
        try:
            # Try to get collection info
            from qdrant_client import QdrantClient
            client = QdrantClient(
                url=qdrant_url, 
                api_key=qdrant_api_key if qdrant_api_key else None,
                https=True if ':443' in qdrant_url else None
            )
            collection_info = client.get_collection(collection_name)
            points_count = collection_info.points_count
            qdrant_status = f"✅ Connected ({points_count} vectors)"
        except:
            qdrant_status = "✅ Connected (info unavailable)"
    
    status_data = {
        "Component": [
            "OpenAI API", 
            "ONNX Model", 
            "Vectorstore", 
            "Qdrant Collection",
            "RAG Chains"
        ],
        "Status": [
            "✅ Connected" if st.session_state.openai_api_key else "❌ Not configured",
            "✅ Ready" if os.path.exists(model_path) else "❌ Model not found",
            "✅ Connected" if vectorstore else "❌ Not connected",
            qdrant_status,
            "✅ Ready" if vectorstore else "❌ Not available"
        ]
    }
    st.table(pd.DataFrame(status_data))
    
    st.info(f"🔗 Qdrant URL: {qdrant_url}")
    st.info(f"📦 Collection: {collection_name}")
    if ':443' in qdrant_url:
        st.info("🔒 Using HTTPS (port 443)")

# Footer
st.markdown("---")
st.markdown("*Powered by ONNX, LangChain, Qdrant Cloud, and OpenAI*")
st.caption("💡 This app connects to your existing Qdrant vectorstore without re-embedding documents.")