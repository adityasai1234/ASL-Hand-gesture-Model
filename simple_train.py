import os
import cv2
import numpy as np
import pickle
import random

# Add kagglehub import
import kagglehub

class SimpleHandGestureTrainer:
    def __init__(self, dataset_path, model_save_path="hand_gesture_model.pkl"):
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path
        
        # Initialize MediaPipe Hands
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
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
        """Prepare dataset from local images"""
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
        
        return X, y
    
    def train_simple_model(self):
        """Train a simple nearest-neighbor model"""
        print("Preparing dataset...")
        X, y = self.prepare_dataset()
        
        if len(X) == 0:
            print("No valid hand landmarks extracted. Check your dataset.")
            return False
            
        print(f"Dataset prepared: {len(X)} samples with {len(X[0])} features")
        
        # Split dataset into train and test (80/20 split)
        combined = list(zip(X, y))
        random.shuffle(combined)
        
        split_idx = int(len(combined) * 0.8)
        train_data = combined[:split_idx]
        test_data = combined[split_idx:]
        
        X_train, y_train = zip(*train_data) if train_data else ([], [])
        X_test, y_test = zip(*test_data) if test_data else ([], [])
        
        # Create a simple model (store all training examples)
        model = {
            'X_train': X_train,
            'y_train': y_train
        }
        
        # Evaluate model using nearest neighbor approach
        correct = 0
        for i, test_features in enumerate(X_test):
            predictions = []
            for train_features in X_train:
                # Calculate Euclidean distance
                distance = np.sqrt(np.sum([(a - b) ** 2 for a, b in zip(test_features, train_features)]))
                predictions.append(distance)
            
            # Get the index of the closest training example
            closest_idx = np.argmin(predictions)
            predicted_label = y_train[closest_idx]
            
            if predicted_label == y_test[i]:
                correct += 1
        
        # Calculate accuracy
        accuracy = correct / len(X_test) if len(X_test) > 0 else 0
        print(f"Model accuracy: {accuracy:.4f}")
        
        # Save model
        with open(self.model_save_path, 'wb') as f:
            pickle.dump(model, f)
            
        print(f"Model saved to {self.model_save_path}")
        return True

def main():
    # Download ASL dataset using kagglehub
    print("Downloading American Sign Language dataset...")
    try:
        dataset_path = kagglehub.dataset_download("kapillondhe/american-sign-language")
        print(f"Dataset downloaded to: {dataset_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Using local dataset path instead.")
        dataset_path = "hand_gesture_dataset"
        
        # Create dataset structure if it doesn't exist
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path, exist_ok=True)
            gestures = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            
            for gesture in gestures:
                gesture_path = os.path.join(dataset_path, gesture)
                os.makedirs(gesture_path, exist_ok=True)
                
                # Create a README file explaining how to add images
                with open(os.path.join(gesture_path, 'README.txt'), 'w') as f:
                    f.write(f"Place {gesture} sign language images in this folder.\n")
                    f.write("Supported formats: .jpg, .jpeg, .png\n")
            
            print(f"Created sample dataset structure at {dataset_path}")
            print("Please add sign language images to each folder before training.")
            return
    
    # Check if dataset folders have images
    has_images = False
    for gesture_class in os.listdir(dataset_path):
        gesture_path = os.path.join(dataset_path, gesture_class)
        if os.path.isdir(gesture_path):
            for file in os.listdir(gesture_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    has_images = True
                    break
            if has_images:
                break
    
    if not has_images:
        print("No images found in the dataset folders.")
        print("Please add hand gesture images to the folders before training.")
        return
    
    # Initialize trainer
    trainer = SimpleHandGestureTrainer(dataset_path)
    
    # Train model
    trainer.train_simple_model()
    
    print("\nTraining complete! You can now use the trained model with your hand gesture detection system.")

if __name__ == "__main__":
    main()