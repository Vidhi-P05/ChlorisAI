"""
Machine Learning Model for Plant Species Classification
Uses TensorFlow/Keras with pretrained CNN (EfficientNet)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import os
import json

class PlantSpeciesClassifier:
    """
    Plant species classifier using a pretrained CNN model
    """
    
    def __init__(self, model_path=None, class_mapping_path=None):
        """
        Initialize the classifier
        
        Args:
            model_path: Path to fine-tuned model (if available)
            class_mapping_path: Path to JSON file mapping class indices to species names
        """
        self.model = None
        self.class_mapping = {}
        self.confidence_threshold = 0.5
        self.alternative_threshold = 0.3
        
        # Load or create model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.create_pretrained_model()
        
        # Load class mapping if available
        if class_mapping_path and os.path.exists(class_mapping_path):
            self.load_class_mapping(class_mapping_path)
        else:
            # Use default ImageNet classes as placeholder
            # In production, this should be replaced with plant-specific classes
            self.create_default_mapping()
    
    def create_pretrained_model(self):
        """
        Create a pretrained EfficientNet model for transfer learning
        This is a base model that should be fine-tuned on plant datasets
        """
        # Load pretrained EfficientNetB0 (ImageNet weights)
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Add custom classification head
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(1024, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        
        # For now, use ImageNet's 1000 classes as placeholder
        # In production, replace with number of plant species
        predictions = keras.layers.Dense(1000, activation='softmax')(x)
        
        self.model = keras.Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers (unfreeze for fine-tuning)
        for layer in base_model.layers:
            layer.trainable = False
        
        print("Pretrained model loaded (ImageNet weights)")
        print("NOTE: This model should be fine-tuned on plant datasets for production use")
    
    def load_model(self, model_path):
        """Load a fine-tuned model from file"""
        try:
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.create_pretrained_model()
    
    def create_default_mapping(self):
        """
        Create a default class mapping using ImageNet classes
        In production, replace with actual plant species mapping
        """
        # This is a placeholder - in production, load from your plant dataset
        # For demonstration, we'll use some ImageNet classes that might be plants
        self.class_mapping = {
            i: {
                'common_name': f'Plant Species {i}',
                'scientific_name': f'Species_{i}',
                'category': 'Plant'
            }
            for i in range(1000)  # ImageNet has 1000 classes
        }
        
        # Add some realistic plant names for demonstration
        # In production, load from your actual dataset
        plant_related_indices = [0, 1, 2, 3, 4, 5]  # Example indices
        plant_names = [
            {'common_name': 'Rose', 'scientific_name': 'Rosa', 'category': 'Flowering Plant'},
            {'common_name': 'Oak Tree', 'scientific_name': 'Quercus', 'category': 'Tree'},
            {'common_name': 'Sunflower', 'scientific_name': 'Helianthus annuus', 'category': 'Flowering Plant'},
            {'common_name': 'Maple Tree', 'scientific_name': 'Acer', 'category': 'Tree'},
            {'common_name': 'Tulip', 'scientific_name': 'Tulipa', 'category': 'Flowering Plant'},
            {'common_name': 'Fern', 'scientific_name': 'Polypodiopsida', 'category': 'Fern'}
        ]
        
        for idx, name_info in zip(plant_related_indices, plant_names):
            self.class_mapping[idx] = name_info
    
    def load_class_mapping(self, mapping_path):
        """Load class mapping from JSON file"""
        try:
            with open(mapping_path, 'r') as f:
                self.class_mapping = json.load(f)
            print(f"Class mapping loaded from {mapping_path}")
        except Exception as e:
            print(f"Error loading class mapping: {e}")
            self.create_default_mapping()
    
    def predict(self, preprocessed_image):
        """
        Predict plant species from preprocessed image
        
        Args:
            preprocessed_image: Preprocessed image array (224x224x3)
            
        Returns:
            Dictionary with predictions, confidence scores, and alternatives
        """
        if self.model is None:
            return {
                'error': 'Model not loaded',
                'has_plant': False
            }
        
        # Ensure image is in correct format
        if len(preprocessed_image.shape) == 3:
            preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        
        # Get predictions
        predictions = self.model.predict(preprocessed_image, verbose=0)
        predictions = predictions[0]  # Get first (and only) prediction
        
        # Get top predictions
        top_indices = np.argsort(predictions)[::-1][:5]  # Top 5
        top_scores = predictions[top_indices]
        
        # Determine if plant is detected (based on confidence threshold)
        max_confidence = float(np.max(predictions))
        has_plant = max_confidence > self.confidence_threshold
        
        # Get primary prediction
        primary_idx = int(top_indices[0])
        primary_confidence = float(top_scores[0])
        
        primary_species = self.class_mapping.get(primary_idx, {
            'common_name': 'Unknown',
            'scientific_name': 'Unknown',
            'category': 'Unknown'
        })
        
        # Get alternatives (if confidence is below threshold or as top 3)
        alternatives = []
        if primary_confidence < 0.7 or has_plant:  # Show alternatives if low confidence or plant detected
            for idx, score in zip(top_indices[1:4], top_scores[1:4]):  # Next 3
                if float(score) > self.alternative_threshold:
                    species_info = self.class_mapping.get(int(idx), {
                        'common_name': 'Unknown',
                        'scientific_name': 'Unknown',
                        'category': 'Unknown'
                    })
                    alternatives.append({
                        'common_name': species_info['common_name'],
                        'scientific_name': species_info['scientific_name'],
                        'confidence': float(score),
                        'category': species_info.get('category', 'Unknown')
                    })
        
        return {
            'has_plant': has_plant,
            'top_prediction': {
                'common_name': primary_species['common_name'],
                'scientific_name': primary_species['scientific_name'],
                'confidence': primary_confidence,
                'category': primary_species.get('category', 'Unknown')
            },
            'alternatives': alternatives,
            'all_predictions': {
                int(idx): float(score) 
                for idx, score in zip(top_indices, top_scores)
            }
        }
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None

