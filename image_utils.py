"""
Image preprocessing and validation utilities
"""

import numpy as np
from PIL import Image
import io
from tensorflow.keras.applications.efficientnet import preprocess_input

def validate_image(file, allowed_extensions, max_size_mb=16):
    """
    Validate uploaded image file
    
    Args:
        file: File object from Flask request
        allowed_extensions: Set of allowed file extensions
        max_size_mb: Maximum file size in MB
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check filename
    if not file.filename:
        return False
    
    # Check extension
    filename = file.filename.lower()
    if not any(filename.endswith(f'.{ext.lower()}') for ext in allowed_extensions):
        return False
    
    # Check file size
    file.seek(0, io.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    
    max_size_bytes = max_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        return False
    
    # Try to open as image to validate format
    try:
        image = Image.open(io.BytesIO(file.read()))
        image.verify()
        file.seek(0)  # Reset file pointer
        return True
    except Exception:
        return False

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model input
    
    Args:
        image: PIL Image object
        target_size: Target size (width, height)
        
    Returns:
        Preprocessed image array ready for model input
    """
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    image_array = np.array(image, dtype=np.float32)
    
    # Apply EfficientNet preprocessing
    # This includes normalization and channel adjustments
    image_array = preprocess_input(image_array)
    
    return image_array

def load_image_from_bytes(image_bytes):
    """
    Load PIL Image from bytes
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        PIL Image object
    """
    return Image.open(io.BytesIO(image_bytes))

