import numpy as np
from PIL import Image

def preprocess_image(image, img_size=224):
    """Preprocess image for ONNX model"""
    img = image.convert("RGB")
    img_resized = img.resize((img_size, img_size))
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    img_chw = np.transpose(img_np, (2, 0, 1))  # HWC → CHW
    img_input = np.expand_dims(img_chw, axis=0)  # Add batch
    return img_input

def softmax(x):
    """Softmax helper function"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()