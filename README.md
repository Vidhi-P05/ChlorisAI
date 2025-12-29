# Plant Species Detection Web Application

A Flask-based web application that uses AI/ML to identify plant species from uploaded images. The application supports multiple image uploads and provides detailed species identification with confidence scores and alternative predictions.

## Features

- **Multiple Image Upload**: Upload multiple plant images (leaf, flower, whole plant) in a single request
- **AI-Powered Identification**: Uses TensorFlow/Keras with EfficientNetB0 (pretrained on ImageNet)
- **Detailed Results**: 
  - Common and scientific names
  - Confidence scores
  - Top 3 alternative predictions
  - Plant detection verification
- **Modern UI**: Beautiful, responsive web interface with drag-and-drop support
- **Image Validation**: Automatic validation of file format and size

## Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow/Keras with EfficientNetB0
- **Image Processing**: PIL/Pillow
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd AurumReads
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Open your browser and navigate to: `http://localhost:5000`

## Usage

1. **Upload Images**:
   - Click "Choose Files" or drag and drop images onto the upload area
   - Supported formats: JPG, PNG
   - Maximum file size: 16MB per image
   - Multiple images can be uploaded simultaneously

2. **View Results**:
   - Click "Identify Plants" to process the images
   - Results will show:
     - Whether a plant was detected
     - Primary species identification with confidence score
     - Alternative predictions (if confidence is below threshold)

## Model Information

### Current Model

The application currently uses **EfficientNetB0** pretrained on ImageNet as a base model. This is a placeholder model that should be fine-tuned on plant-specific datasets for production use.

### Fine-Tuning for Production

To improve accuracy for plant species identification:

1. **Prepare Plant Dataset**:
   - Use datasets like:
     - [iNaturalist](https://www.inaturalist.org/)
     - [PlantNet](https://plantnet.org/)
     - [PlantCLEF](https://www.imageclef.org/lifeclef/2018/plant)
   
2. **Fine-Tune the Model**:
   - Create a training script to fine-tune the EfficientNetB0 model
   - Replace the classification head with the number of plant species in your dataset
   - Train on your plant dataset

3. **Update Class Mapping**:
   - Create a JSON file mapping class indices to plant species names
   - Update `class_mapping_path` in `ml_model.py`

4. **Save and Load Model**:
   - Save the fine-tuned model
   - Update `model_path` in `PlantSpeciesClassifier` initialization

### Example Fine-Tuning Structure

```python
# In ml_model.py, update the model creation:
# Replace 1000 with your number of plant species
predictions = keras.layers.Dense(num_plant_species, activation='softmax')(x)
```

## Project Structure

```
AurumReads/
├── app.py                 # Main Flask application
├── ml_model.py            # ML model implementation
├── image_utils.py         # Image preprocessing utilities
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Frontend HTML template
├── static/
│   ├── style.css         # CSS styles
│   └── script.js         # Frontend JavaScript
└── uploads/              # Uploaded images (created automatically)
```

## API Endpoints

### `GET /`
- Renders the main web interface

### `POST /upload`
- Uploads and processes plant images
- **Request**: Multipart form data with `images` field (multiple files allowed)
- **Response**: JSON with identification results
  ```json
  {
    "results": [
      {
        "filename": "plant.jpg",
        "has_plant": true,
        "primary_species": {
          "common_name": "Rose",
          "scientific_name": "Rosa",
          "confidence": 0.85,
          "category": "Flowering Plant"
        },
        "alternatives": [...]
      }
    ]
  }
  ```

### `GET /health`
- Health check endpoint
- Returns model status

## Configuration

Edit `app.py` to customize:

- `UPLOAD_FOLDER`: Directory for uploaded images
- `MAX_CONTENT_LENGTH`: Maximum file size (default: 16MB)
- `ALLOWED_EXTENSIONS`: Allowed image formats

Edit `ml_model.py` to customize:

- `confidence_threshold`: Minimum confidence for plant detection (default: 0.5)
- `alternative_threshold`: Minimum confidence for alternative predictions (default: 0.3)

## Development

### Running in Development Mode

The application runs in debug mode by default. For production:

1. Set `debug=False` in `app.py`
2. Use a production WSGI server (e.g., Gunicorn, uWSGI)
3. Configure proper security settings

### Adding New Features

- **Model Improvements**: Modify `ml_model.py` to use different architectures or add post-processing
- **UI Enhancements**: Update `templates/index.html` and `static/style.css`
- **API Extensions**: Add new routes in `app.py`

## Limitations

- Current model uses ImageNet weights (not plant-specific)
- Accuracy will be limited until fine-tuned on plant datasets
- Processing time depends on image size and hardware
- GPU recommended for faster inference

## Future Improvements

- [ ] Fine-tune model on plant-specific datasets
- [ ] Add image preview before upload
- [ ] Implement batch processing for large datasets
- [ ] Add model versioning and A/B testing
- [ ] Implement caching for faster repeated predictions
- [ ] Add user authentication and history
- [ ] Support for more image formats
- [ ] Mobile app integration

## License

This project is open source and available for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Acknowledgments

- TensorFlow/Keras for the ML framework
- EfficientNet architecture by Google Research
- Flask for the web framework

