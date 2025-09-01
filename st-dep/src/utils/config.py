import yaml
import os
import streamlit as st

def load_config(config_path="configs/app_config.yaml"):
    """Load application configuration"""
    default_config = {
        'model': {
            'path': "models/die_crack_classifier.onnx",
            'image_size': 224,
            'classes': ["cracked die at upper left part", "cracked die at lower left part", "cracked die at lower right part", "cracked die at upper right part", "shattered die"]
        },
        'documents': {
            'static_path': "documents/",
            'upload_path': "uploads/",
            'chunk_size': 1000,
            'chunk_overlap': 55
        },
        'rag': {
            'collection_name': "MOSFET_Die_Crack_Analysis",
            'retrieval_k': 10,
            'confidence_threshold': 0.2
        }
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Merge with defaults
                return {**default_config, **config}
    except Exception as e:
        st.warning(f"Could not load config file: {e}. Using defaults.")
    
    return default_config

def check_api_key(api_key):
    """Check if OpenAI API key is provided"""
    return bool(api_key and api_key.strip())