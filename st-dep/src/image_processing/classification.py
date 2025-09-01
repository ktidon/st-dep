import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

@st.cache_resource
def load_onnx_model(model_path):
    """Load ONNX model"""
    try:
        return ort.InferenceSession(model_path)
    except Exception as e:
        st.error(f"Error loading ONNX model: {e}")
        return None

def predict_image_class(session, image, class_names, img_size=224):
    """Predict image class using ONNX model"""
    try:
        img_input = preprocess_image(image, img_size)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_input})
        
        logits = outputs[0][0]  # shape: (num_classes,)
        probs = softmax(logits)
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]
        
        return class_names[pred_class], float(confidence)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None