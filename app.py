"""
Flask Plant Species Detection Application
Main application file for handling image uploads and plant species identification
"""

import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import io

from ml_model import PlantSpeciesClassifier
from image_utils import preprocess_image, validate_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the ML model
classifier = PlantSpeciesClassifier()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    """
    Handle multiple image uploads and return plant species predictions
    """
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    files = request.files.getlist('images')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    results = []
    
    for file in files:
        # Validate file
        if not validate_image(file, app.config['ALLOWED_EXTENSIONS']):
            results.append({
                'filename': file.filename,
                'error': 'Invalid file format or size'
            })
            continue
        
        try:
            # Read image
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Preprocess image
            processed_image = preprocess_image(image)
            
            # Get predictions
            predictions = classifier.predict(processed_image)
            
            # Format results
            result = {
                'filename': file.filename,
                'predictions': predictions,
                'has_plant': predictions['has_plant'],
                'primary_species': predictions['top_prediction'],
                'alternatives': predictions['alternatives'] if predictions.get('alternatives') else []
            }
            
            results.append(result)
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': f'Processing error: {str(e)}'
            })
    
    return jsonify({'results': results}), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier.is_loaded()
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

