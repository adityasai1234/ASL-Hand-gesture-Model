import os
import cv2
import numpy as np
import pickle
import random

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
            
        # Normalize landmarks to improve accuracy
        landmarks = self._normalize_landmarks(landmarks)
            
        return landmarks
    
    def _normalize_landmarks(self, landmarks):
        """Normalize landmarks to make them scale and position invariant"""
        # Reshape to get x, y, z coordinates
        coords = np.array(landmarks).reshape(-1, 3)
        
        # Calculate centroid
        centroid = np.mean(coords, axis=0)
        
        # Center the landmarks
        centered = coords - centroid
        
        # Scale to unit size
        max_distance = np.max(np.linalg.norm(centered, axis=1))
        if max_distance > 0:
            normalized = centered / max_distance
        else:
            normalized = centered
            
        # Flatten back to 1D array
        return normalized.flatten().tolist()
    
    def prepare_dataset(self):
        """Prepare dataset from local images with data augmentation"""
        X = []  # Features (landmarks)
        y = []  # Labels (gesture classes)
        
        # Iterate through gesture folders
        for gesture_class in os.listdir(self.dataset_path):
            gesture_path = os.path.join(self.dataset_path, gesture_class)
            
            if not os.path.isdir(gesture_path):
                continue
                
            print(f"Processing gesture: {gesture_class}")
            
            # Process each image in the gesture folder
            image_count = 0
            for image_file in os.listdir(gesture_path):
                if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                image_path = os.path.join(gesture_path, image_file)
                landmarks = self.extract_landmarks(image_path)
                
                if landmarks:
                    X.append(landmarks)
                    y.append(gesture_class)
                    image_count += 1
                    
                    # Apply data augmentation for small variations
                    for _ in range(3):  # Create 3 augmented samples per image
                        augmented = self._augment_landmarks(landmarks)
                        X.append(augmented)
                        y.append(gesture_class)
            
            print(f"  - Processed {image_count} images for {gesture_class}")
        
        return X, y
    
    def _augment_landmarks(self, landmarks):
        """Apply small random variations to landmarks for data augmentation"""
        # Convert to numpy array
        landmarks_array = np.array(landmarks)
        
        # Add small random noise (±2% variation)
        noise = np.random.normal(0, 0.02, landmarks_array.shape)
        augmented = landmarks_array + noise
        
        return augmented.tolist()
    
    def train_simple_model(self, epochs=20, batch_size=1000, early_stop=True):
        """Train a simple nearest-neighbor model with weighted voting over multiple epochs"""
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
        
        # Convert to numpy arrays for easier manipulation
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        
        # Initialize best model
        best_accuracy = 0
        best_model = None
        no_improvement_count = 0
        
        # Train over multiple epochs
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            # Shuffle training data for each epoch
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Apply feature selection - focus on most important landmarks
            feature_weights = np.ones(X_train.shape[1])
            
            # Give more weight to fingertips (landmarks 4, 8, 12, 16, 20)
            fingertip_indices = []
            for tip in [4, 8, 12, 16, 20]:
                fingertip_indices.extend([tip*3, tip*3+1, tip*3+2])
            
            feature_weights[fingertip_indices] = 1.5 + 0.1 * np.random.rand(len(fingertip_indices))
            
            # Apply weights to features - vectorized operation
            X_train_weighted = X_train_shuffled * feature_weights
            X_test_weighted = X_test * feature_weights
            
            # Create current epoch model
            current_model = {
                'X_train': X_train_weighted,
                'y_train': y_train_shuffled,
                'feature_weights': feature_weights
            }
            
            # Evaluate model using k-nearest neighbors approach with batching
            k = 5  # Number of neighbors to consider
            correct = 0
            
            # Process test samples in batches to avoid memory issues
            num_test_batches = (len(X_test_weighted) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_test_batches):
                print(f"  Processing test batch {batch_idx+1}/{num_test_batches}...", end="\r")
                
                # Get current batch of test samples
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(X_test_weighted))
                X_test_batch = X_test_weighted[start_idx:end_idx]
                
                # Process each test sample in the batch
                for i, test_features in enumerate(X_test_batch):
                    # Calculate distances to all training examples
                    distances = np.sqrt(np.sum((X_train_weighted - test_features)**2, axis=1))
                    
                    # Get the k closest training examples
                    k_nearest_indices = np.argsort(distances)[:k]
                    k_nearest_labels = y_train_shuffled[k_nearest_indices]
                    k_nearest_distances = distances[k_nearest_indices]
                    
                    # Weight votes by inverse distance
                    label_weights = {}
                    for label, dist in zip(k_nearest_labels, k_nearest_distances):
                        weight = 1.0 / (dist + 1e-5)
                        if label in label_weights:
                            label_weights[label] += weight
                        else:
                            label_weights[label] = weight
                    
                    # Get the label with highest weighted votes
                    predicted_label = max(label_weights, key=label_weights.get)
                    
                    # Check if prediction is correct
                    true_idx = start_idx + i
                    if predicted_label == y_test[true_idx]:
                        correct += 1
            
            print("                                                  ", end="\r")  # Clear progress line
            
            # Calculate accuracy
            accuracy = correct / len(X_test) if len(X_test) > 0 else 0
            print(f"Epoch {epoch} accuracy: {accuracy:.4f}")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = current_model
                print(f"New best model with accuracy: {best_accuracy:.4f}")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                print(f"No improvement for {no_improvement_count} epochs")
                
            # Early stopping if accuracy is very high or no improvement for 3 epochs
            if early_stop and (accuracy > 0.999 or no_improvement_count >= 3):
                print(f"\nEarly stopping at epoch {epoch} with accuracy {accuracy:.4f}")
                break
        
        print(f"\nTraining complete. Best model accuracy: {best_accuracy:.4f}")
        
        # Save best model
        with open(self.model_save_path, 'wb') as f:
            pickle.dump(best_model, f)
            
        print(f"Best model saved to {self.model_save_path}")
        return True

def create_sample_dataset(base_path):
    """Create a sample dataset structure for testing"""
    gestures = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    for gesture in gestures:
        gesture_path = os.path.join(base_path, gesture)
        os.makedirs(gesture_path, exist_ok=True)
        
        # Create a README file explaining how to add images
        with open(os.path.join(gesture_path, 'README.txt'), 'w') as f:
            f.write(f"Place {gesture} sign language images in this folder.\n")
            f.write("Supported formats: .jpg, .jpeg, .png\n")
    
    print(f"Created sample dataset structure at {base_path}")
    print("Please add sign language images to each folder before training.")

def main():
    dataset_path = "hand_gesture_dataset"
    
    # Create dataset structure if it doesn't exist
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)
        create_sample_dataset(dataset_path)
        print("\nPlease add images to the dataset folders and run this script again.")
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