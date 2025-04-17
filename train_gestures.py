import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class HandGestureTrainer:
    def __init__(self, dataset_path, model_save_path="hand_gesture_model.pkl"):
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Initialize model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def extract_landmarks(self, image_path):
        """Extract hand landmarks from an image"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return None
            
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.hands.process(image_rgb)
        
        # Check if hand landmarks were detected
        if not results.multi_hand_landmarks:
            return None
            
        # Extract landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        
        # Flatten landmarks into a 1D array
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
            
        return landmarks
    
    def prepare_dataset(self):
        """Prepare dataset from Kaggle images"""
        X = []  # Features (landmarks)
        y = []  # Labels (gesture classes)
        
        # Iterate through gesture folders
        for gesture_class in os.listdir(self.dataset_path):
            gesture_path = os.path.join(self.dataset_path, gesture_class)
            
            if not os.path.isdir(gesture_path):
                continue
                
            print(f"Processing gesture: {gesture_class}")
            
            # Process each image in the gesture folder
            for image_file in os.listdir(gesture_path):
                if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                image_path = os.path.join(gesture_path, image_file)
                landmarks = self.extract_landmarks(image_path)
                
                if landmarks:
                    X.append(landmarks)
                    y.append(gesture_class)
        
        return np.array(X), np.array(y)
    
    def train_model(self):
        """Train the hand gesture recognition model"""
        print("Preparing dataset...")
        X, y = self.prepare_dataset()
        
        if len(X) == 0:
            print("No valid hand landmarks extracted. Check your dataset.")
            return False
            
        print(f"Dataset prepared: {len(X)} samples with {len(X[0])} features")
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        with open(self.model_save_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        print(f"Model saved to {self.model_save_path}")
        return True
    
    def load_model(self):
        """Load a trained model"""
        if os.path.exists(self.model_save_path):
            with open(self.model_save_path, 'rb') as f:
                self.model = pickle.load(f)
            return True
        return False

def download_kaggle_dataset(dataset_name, save_path):
    """Download a dataset from Kaggle"""
    try:
        import kaggle
        print(f"Downloading dataset {dataset_name}...")
        kaggle.api.dataset_download_files(dataset_name, path=save_path, unzip=True)
        print(f"Dataset downloaded to {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTo download from Kaggle, you need to:")
        print("1. Install kaggle package: pip install kaggle")
        print("1150187367ca32ef3c598303f65b9eca")
        print(r"C:\Users\adity\Downloads\kaggle.json")  # Fixed with raw string
        return False

def main():
    # Example Kaggle dataset for hand gestures
    # You can replace this with any hand gesture dataset from Kaggle
    kaggle_dataset = "gti-upm/leapgestrecog"
    dataset_path = "hand_gesture_dataset"
    
    # Download dataset if needed
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)
        success = download_kaggle_dataset(kaggle_dataset, dataset_path)
        if not success:
            print("Please download a hand gesture dataset manually and place it in the 'hand_gesture_dataset' folder")
            return
    
    # Initialize trainer
    trainer = HandGestureTrainer(dataset_path)
    
    # Train model
    trainer.train_model()
    
    print("\nTraining complete! You can now use the trained model with your hand gesture detection system.")
    print("To integrate with your existing code, see the integration_example.py file.")

if __name__ == "__main__":
    main()