# Plant Disease Detection using CNN - Complete Implementation
# PlantVillage Dataset Classification

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import cv2
from PIL import Image
import zipfile
import requests
from pathlib import Path
import pickle

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class PlantDiseaseDetector:
    def __init__(self, img_size=(224, 224), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.class_names = []
        self.history = None
        
    def download_dataset(self, data_dir="plant_village_data"):
        """Download and extract PlantVillage dataset"""
        print("Note: This is a simplified dataset loader.")
        print("For the complete PlantVillage dataset, download from:")
        print("https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        
        # Create sample directory structure for demonstration
        os.makedirs(f"{data_dir}/train", exist_ok=True)
        os.makedirs(f"{data_dir}/validation", exist_ok=True)
        
        # Sample class names from PlantVillage dataset
        self.class_names = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        
        print(f"Dataset structure created in {data_dir}/")
        print(f"Total classes: {len(self.class_names)}")
        return data_dir
    
    def create_data_generators(self, data_dir):
        """Create data generators with augmentation"""
        # Data augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Only rescaling for validation
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Create generators (assuming you have the actual dataset)
        train_generator = train_datagen.flow_from_directory(
            f"{data_dir}/train",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            f"{data_dir}/train",  # Using same directory with split
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def build_cnn_model(self, num_classes):
        """Build CNN model architecture"""
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fifth Convolutional Block
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def build_transfer_learning_model(self, num_classes):
        """Build model using transfer learning with EfficientNetB0"""
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, model, learning_rate=0.001):
        """Compile the model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_5_accuracy']
        )
        
        return model
    
    def train_model(self, train_gen, val_gen, epochs=50, model_type='cnn'):
        """Train the model"""
        num_classes = len(self.class_names)
        
        # Build model
        if model_type == 'transfer':
            self.model = self.build_transfer_learning_model(num_classes)
        else:
            self.model = self.build_cnn_model(num_classes)
        
        # Compile model
        self.model = self.compile_model(self.model)
        
        print("Model Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            ),
            keras.callbacks.ModelCheckpoint(
                'best_plant_disease_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        print(f"Training {model_type} model for {epochs} epochs...")
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-5 Accuracy
        if 'top_5_accuracy' in self.history.history:
            axes[1, 0].plot(self.history.history['top_5_accuracy'], label='Training Top-5 Accuracy')
            axes[1, 0].plot(self.history.history['val_top_5_accuracy'], label='Validation Top-5 Accuracy')
            axes[1, 0].set_title('Model Top-5 Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-5 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning Rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self, test_gen):
        """Evaluate model performance"""
        if self.model is None:
            print("No model available. Train the model first.")
            return
        
        # Evaluate
        test_loss, test_accuracy, test_top5_accuracy = self.model.evaluate(test_gen, verbose=1)
        
        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Top-5 Accuracy: {test_top5_accuracy:.4f}")
        
        # Predictions for confusion matrix
        predictions = self.model.predict(test_gen)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_gen.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, 
                                  target_names=list(test_gen.class_indices.keys())))
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(test_gen.class_indices.keys()),
                   yticklabels=list(test_gen.class_indices.keys()))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return test_accuracy
    
    def preprocess_image(self, image_path):
        """Preprocess single image for prediction"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict_disease(self, image_path, top_k=5):
        """Predict disease for a single image"""
        if self.model is None:
            print("No model available. Train the model first.")
            return None
        
        # Preprocess image
        img = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img)
        probabilities = predictions[0]
        
        # Get top k predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        results = []
        for i in top_indices:
            disease_name = self.class_names[i]
            confidence = probabilities[i] * 100
            results.append({
                'disease': disease_name,
                'confidence': confidence
            })
        
        return results
    
    def save_model(self, filepath='plant_disease_model.h5'):
        """Save the trained model"""
        if self.model is None:
            print("No model to save. Train the model first.")
            return
        
        self.model.save(filepath)
        
        # Save class names
        with open('class_names.pkl', 'wb') as f:
            pickle.dump(self.class_names, f)
        
        print(f"Model saved to {filepath}")
        print("Class names saved to class_names.pkl")
    
    def load_model(self, filepath='plant_disease_model.h5'):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        
        # Load class names
        try:
            with open('class_names.pkl', 'rb') as f:
                self.class_names = pickle.load(f)
        except FileNotFoundError:
            print("Warning: class_names.pkl not found. Using default class names.")
        
        print(f"Model loaded from {filepath}")
    
    def create_gradcam_heatmap(self, image_path, layer_name=None):
        """Create Grad-CAM heatmap for model interpretability"""
        if self.model is None:
            print("No model available.")
            return None
        
        # Get the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
        
        # Preprocess image
        img = self.preprocess_image(image_path)
        
        # Create a model that maps the input image to the activations of the last conv layer
        grad_model = tf.keras.models.Model([self.model.inputs], 
                                         [self.model.get_layer(layer_name).output, self.model.output])
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            predicted_class = tf.argmax(predictions[0])
            class_channel = predictions[:, predicted_class]
        
        # Compute gradients of the class with respect to the feature maps
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Pool gradients over spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps by the gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()

# Demo usage and testing functions
def create_sample_training_script():
    """Sample script to demonstrate usage"""
    print("=" * 60)
    print("PLANT DISEASE DETECTION SYSTEM - DEMO")
    print("=" * 60)
    
    # Initialize detector
    detector = PlantDiseaseDetector(img_size=(224, 224), batch_size=32)
    
    # Note: In real implementation, you would download the actual dataset
    print("\n1. Dataset Preparation:")
    print("To use this system with real data:")
    print("- Download PlantVillage dataset from Kaggle")
    print("- Extract to 'plant_village_data/' directory")
    print("- Organize as train/validation folders")
    
    print(f"\n2. Model will classify {len(detector.class_names)} plant disease classes")
    
    print("\n3. Training Process:")
    print("- Data augmentation applied")
    print("- CNN or Transfer Learning architecture")
    print("- Early stopping and learning rate scheduling")
    print("- Model checkpointing")
    
    print("\n4. Features Available:")
    print("- Training with data augmentation")
    print("- Model evaluation with metrics")
    print("- Single image prediction")
    print("- Grad-CAM visualization")
    print("- Model saving/loading")
    
    return detector

# Example usage
if __name__ == "__main__":
    # Create detector instance
    detector = create_sample_training_script()
    
    # Example of how to use (requires actual dataset)
    """
    # 1. Setup data
    data_dir = detector.download_dataset()
    train_gen, val_gen = detector.create_data_generators(data_dir)
    
    # 2. Train model
    history = detector.train_model(train_gen, val_gen, epochs=50, model_type='transfer')
    
    # 3. Plot training history
    detector.plot_training_history()
    
    # 4. Evaluate model
    test_accuracy = detector.evaluate_model(val_gen)
    
    # 5. Save model
    detector.save_model('plant_disease_model.h5')
    
    # 6. Make predictions
    results = detector.predict_disease('path/to/test/image.jpg')
    print("Prediction results:", results)
    """
    
    print("\n" + "=" * 60)
    print("Setup complete! Uncomment the usage example to train with real data.")
    print("=" * 60)