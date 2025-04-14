
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

class HandGestureDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_path="hand_gesture_model.pkl"):
        # Initialize MediaPipe Hands solution with improved parameters
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1  # Use more accurate model (0, 1, or 2)
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize landmark smoothing
        self.smoothing_factor = 0.5  # Adjust between 0 (no smoothing) and 1 (max smoothing)
        self.previous_landmarks = []
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
        
        # Load ASL recognition model if it exists
        self.asl_model = None
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.asl_model = pickle.load(f)
                print(f"ASL recognition model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading ASL model: {e}")
        
        # Define gesture recognition thresholds
        self.gestures = {
            'open_palm': self._is_open_palm,
            'closed_fist': self._is_closed_fist,
            'pointing': self._is_pointing,
            'thumbs_up': self._is_thumbs_up,
            'peace': self._is_peace_sign
        }
    
    # Add ASL recognition method
    def recognize_asl(self, landmarks):
        """Recognize ASL letter from hand landmarks"""
        if self.asl_model is None:
            return None
        
        try:
            # Normalize landmarks (same as in training)
            normalized_landmarks = self._normalize_landmarks(landmarks)
            
            # Get training data and feature weights from model
            X_train = np.array(self.asl_model['X_train'])
            y_train = np.array(self.asl_model['y_train'])
            
            # Check if dimensions match
            if len(normalized_landmarks) != X_train.shape[1]:
                # Trim or pad to match dimensions
                if len(normalized_landmarks) > X_train.shape[1]:
                    normalized_landmarks = normalized_landmarks[:X_train.shape[1]]
                else:
                    normalized_landmarks = np.pad(normalized_landmarks, 
                                                 (0, X_train.shape[1] - len(normalized_landmarks)),
                                                 'constant')
            
            feature_weights = self.asl_model.get('feature_weights', np.ones(len(normalized_landmarks)))
            
            # Check if training data is empty
            if len(X_train) == 0 or len(y_train) == 0:
                return "No Data"
            
            # Apply feature weights to the test landmarks
            weighted_landmarks = np.array(normalized_landmarks) * feature_weights
            
            # Calculate distances to all training examples
            distances = np.sqrt(np.sum((X_train - weighted_landmarks)**2, axis=1))
            
            # Get the k nearest neighbors
            k = min(7, len(y_train))  # Ensure k is not larger than the dataset
            k_nearest_indices = np.argsort(distances)[:k]
            
            # Get labels of k nearest neighbors
            k_nearest_labels = y_train[k_nearest_indices]
            k_nearest_distances = distances[k_nearest_indices]
            
            # Weight votes by inverse distance
            label_weights = {}
            for label, dist in zip(k_nearest_labels, k_nearest_distances):
                # Use squared inverse distance for stronger weighting
                weight = 1.0 / (dist**2 + 1e-5)
                if label in label_weights:
                    label_weights[label] += weight
                else:
                    label_weights[label] = weight
            
            # Get the label with highest weighted votes
            if not label_weights:
                return None
                
            predicted_letter = max(label_weights, key=label_weights.get)
            
            # Calculate confidence score
            total_weight = sum(label_weights.values())
            confidence = label_weights[predicted_letter] / total_weight if total_weight > 0 else 0
            
            # Only return prediction if confidence is high enough
            if confidence > 0.6:  # Minimum 60% confidence
                return predicted_letter
            else:
                return None
                
        except Exception as e:
            print(f"Error in ASL recognition: {e}")
            return None
    
    def _normalize_landmarks(self, landmarks):
        """Normalize landmarks to make them scale and position invariant"""
        # Extract x, y, z coordinates
        coords = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        
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
        
        # Add relative angles between fingers as additional features
        # This helps distinguish similar hand shapes
        normalized_flat = normalized.flatten()
        
        # Add finger joint angles
        angles = self._calculate_finger_angles(coords)
        
        # Combine normalized coordinates with angles
        combined_features = np.concatenate([normalized_flat, angles])
            
        return combined_features
    
    def _calculate_finger_angles(self, coords):
        """Calculate angles between finger joints to improve recognition"""
        angles = []
        
        # Define finger joints (wrist, knuckle, PIP, DIP, tip)
        fingers = [
            [0, 1, 2, 3, 4],    # Thumb
            [0, 5, 6, 7, 8],    # Index
            [0, 9, 10, 11, 12], # Middle
            [0, 13, 14, 15, 16], # Ring
            [0, 17, 18, 19, 20]  # Pinky
        ]
        
        # Calculate angles for each finger
        for finger in fingers:
            for i in range(1, len(finger)-1):
                # Get three consecutive joints
                p1 = coords[finger[i-1]]
                p2 = coords[finger[i]]
                p3 = coords[finger[i+1]]
                
                # Calculate vectors
                v1 = p1 - p2
                v2 = p3 - p2
                
                # Calculate angle using dot product
                dot = np.dot(v1, v2)
                norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                
                # Avoid division by zero
                if norm < 1e-6:
                    angle = 0
                else:
                    # Clamp to avoid numerical errors
                    cos_angle = np.clip(dot / norm, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                
                angles.append(angle)
        
        return np.array(angles)
    
    def detect_gestures(self, image):
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = self.hands.process(image_rgb)
        
        # Initialize list to store hand landmarks and gestures
        hands_data = []
        
        # Check if hands were detected
        if results.multi_hand_landmarks:
            # Reset previous landmarks if number of hands changed
            if len(self.previous_landmarks) != len(results.multi_hand_landmarks):
                self.previous_landmarks = [None] * len(results.multi_hand_landmarks)
            
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand label (left or right)
                hand_label = "Right" if results.multi_handedness[hand_idx].classification[0].label == "Right" else "Left"
                
                # Extract landmark coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                # Apply smoothing if previous landmarks exist
                if self.previous_landmarks[hand_idx] is not None:
                    landmarks = self._smooth_landmarks(landmarks, self.previous_landmarks[hand_idx])
                
                # Store current landmarks for next frame
                self.previous_landmarks[hand_idx] = landmarks
                
                # Detect gestures
                detected_gestures = []
                for gesture_name, gesture_func in self.gestures.items():
                    if gesture_func(landmarks):
                        detected_gestures.append(gesture_name)
                
                # Recognize ASL letter if model is loaded
                asl_letter = self.recognize_asl(landmarks)
                
                # Store hand data
                hands_data.append({
                    'hand_label': hand_label,
                    'landmarks': landmarks,
                    'gestures': detected_gestures,
                    'asl_letter': asl_letter
                })
        else:
            # Reset previous landmarks if no hands detected
            self.previous_landmarks = []
        
        return hands_data
    
    def _smooth_landmarks(self, current_landmarks, previous_landmarks):
        """Apply temporal smoothing to landmarks to reduce jitter"""
        smoothed_landmarks = []
        
        for i, landmark in enumerate(current_landmarks):
            # Get previous landmark
            prev = previous_landmarks[i]
            
            # Apply exponential smoothing
            smoothed = {
                'x': prev['x'] * self.smoothing_factor + landmark['x'] * (1 - self.smoothing_factor),
                'y': prev['y'] * self.smoothing_factor + landmark['y'] * (1 - self.smoothing_factor),
                'z': prev['z'] * self.smoothing_factor + landmark['z'] * (1 - self.smoothing_factor)
            }
            
            smoothed_landmarks.append(smoothed)
            
        return smoothed_landmarks
    
    def draw_landmarks(self, image, hands_data, results):
        # Create a copy of the image
        output_image = image.copy()
        
        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks with custom styling for better visibility
                self.mp_drawing.draw_landmarks(
                    output_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 255, 255), thickness=5, circle_radius=5),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(255, 255, 0), thickness=3)
                )
        
        # Add gesture labels and ASL letters
        for idx, hand_data in enumerate(hands_data):
            # Get the wrist position for text placement
            wrist = hand_data['landmarks'][0]
            wrist_pos = (
                int(wrist['x'] * output_image.shape[1]),
                int(wrist['y'] * output_image.shape[0])
            )
            
            # Create a semi-transparent background for text
            text_bg_color = (0, 0, 0)
            text_bg_alpha = 0.5
            text_padding = 10
            text_y_offset = 30
            
            # Add ASL letter if detected
            if hand_data.get('asl_letter'):
                # Prepare text
                asl_text = f"ASL: {hand_data['asl_letter']}"
                text_size = cv2.getTextSize(
                    asl_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                
                # Text position
                asl_text_pos = (wrist_pos[0], wrist_pos[1] - text_y_offset)
                
                # Draw background rectangle
                bg_rect_pt1 = (
                    asl_text_pos[0] - text_padding, 
                    asl_text_pos[1] - text_size[1] - text_padding
                )
                bg_rect_pt2 = (
                    asl_text_pos[0] + text_size[0] + text_padding, 
                    asl_text_pos[1] + text_padding
                )
                
                # Create semi-transparent overlay
                overlay = output_image.copy()
                cv2.rectangle(overlay, bg_rect_pt1, bg_rect_pt2, text_bg_color, -1)
                output_image = cv2.addWeighted(
                    overlay, text_bg_alpha, output_image, 1 - text_bg_alpha, 0)
                
                # Draw text
                cv2.putText(
                    output_image,
                    asl_text,
                    asl_text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),  # Yellow color for better visibility
                    2
                )
                text_y_offset += 40  # Increased spacing
            
            # Add gestures
            if hand_data['gestures']:
                for i, gesture in enumerate(hand_data['gestures']):
                    # Prepare text
                    gesture_text = f"{hand_data['hand_label']}: {gesture}"
                    text_size = cv2.getTextSize(
                        gesture_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    # Text position
                    text_pos = (wrist_pos[0], wrist_pos[1] - text_y_offset + (i * 35))
                    
                    # Draw background rectangle
                    bg_rect_pt1 = (
                        text_pos[0] - text_padding, 
                        text_pos[1] - text_size[1] - text_padding
                    )
                    bg_rect_pt2 = (
                        text_pos[0] + text_size[0] + text_padding, 
                        text_pos[1] + text_padding
                    )
                    
                    # Create semi-transparent overlay
                    overlay = output_image.copy()
                    cv2.rectangle(overlay, bg_rect_pt1, bg_rect_pt2, text_bg_color, -1)
                    output_image = cv2.addWeighted(
                        overlay, text_bg_alpha, output_image, 1 - text_bg_alpha, 0)
                    
                    # Draw text
                    cv2.putText(
                        output_image,
                        gesture_text,
                        text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),  # Green color
                        2
                    )
        
        return output_image

    def detect_faces(self, image):
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect faces
        results = self.face_detection.process(image_rgb)
        
        # Initialize list to store face detections
        face_detections = []
        
        # Check if faces were detected
        if results.detections:
            for detection in results.detections:
                bounding_box = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                # Convert relative coordinates to absolute
                x = int(bounding_box.xmin * w)
                y = int(bounding_box.ymin * h)
                width = int(bounding_box.width * w)
                height = int(bounding_box.height * h)
                
                # Store face data
                face_detections.append({
                    'box': (x, y, width, height),
                    'confidence': detection.score[0]
                })
        
        return face_detections
    
    def draw_faces(self, image, face_detections):
        # Create a copy of the image
        output_image = image.copy()
        
        # Draw face bounding boxes
        for detection in face_detections:
            box = detection['box']
            confidence = detection['confidence']
            
            # Extract coordinates
            x, y, width, height = box
            
            # Draw rectangle
            cv2.rectangle(output_image, 
                         (x, y), 
                         (x + width, y + height), 
                         (0, 0, 255), 2)
            
            # Add confidence text
            text = f"Face: {confidence:.2f}"
            cv2.putText(output_image, text, 
                       (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 0, 255), 2)
        
        return output_image

    # Gesture detection methods
    def _is_open_palm(self, landmarks):
        # Check if all fingers are extended
        thumb_tip = landmarks[4]['y']
        thumb_ip = landmarks[3]['y']
        
        index_tip = landmarks[8]['y']
        index_pip = landmarks[6]['y']
        
        middle_tip = landmarks[12]['y']
        middle_pip = landmarks[10]['y']
        
        ring_tip = landmarks[16]['y']
        ring_pip = landmarks[14]['y']
        
        pinky_tip = landmarks[20]['y']
        pinky_pip = landmarks[18]['y']
        
        # Check if all fingertips are above their PIPs (extended)
        return (thumb_tip < thumb_ip and 
                index_tip < index_pip and 
                middle_tip < middle_pip and 
                ring_tip < ring_pip and 
                pinky_tip < pinky_pip)
    
    def _is_closed_fist(self, landmarks):
        # Check if all fingers are curled (fingertips below PIPs)
        thumb_tip = landmarks[4]['y']
        thumb_ip = landmarks[3]['y']
        
        index_tip = landmarks[8]['y']
        index_pip = landmarks[6]['y']
        
        middle_tip = landmarks[12]['y']
        middle_pip = landmarks[10]['y']
        
        ring_tip = landmarks[16]['y']
        ring_pip = landmarks[14]['y']
        
        pinky_tip = landmarks[20]['y']
        pinky_pip = landmarks[18]['y']
        
        # Check if all fingertips are below their PIPs (curled)
        return (thumb_tip > thumb_ip and 
                index_tip > index_pip and 
                middle_tip > middle_pip and 
                ring_tip > ring_pip and 
                pinky_tip > pinky_pip)
    
    def _is_pointing(self, landmarks):
        # Check if index finger is extended but others are curled
        index_tip = landmarks[8]['y']
        index_pip = landmarks[6]['y']
        
        middle_tip = landmarks[12]['y']
        middle_pip = landmarks[10]['y']
        
        ring_tip = landmarks[16]['y']
        ring_pip = landmarks[14]['y']
        
        pinky_tip = landmarks[20]['y']
        pinky_pip = landmarks[18]['y']
        
        return (index_tip < index_pip and  # Index extended
                middle_tip > middle_pip and  # Middle curled
                ring_tip > ring_pip and  # Ring curled
                pinky_tip > pinky_pip)  # Pinky curled
    
    def _is_thumbs_up(self, landmarks):
        # Check if thumb is extended upward and other fingers are curled
        thumb_tip_y = landmarks[4]['y']
        thumb_ip_y = landmarks[3]['y']
        thumb_tip_x = landmarks[4]['x']
        thumb_ip_x = landmarks[3]['x']
        
        index_tip = landmarks[8]['y']
        index_pip = landmarks[6]['y']
        
        middle_tip = landmarks[12]['y']
        middle_pip = landmarks[10]['y']
        
        ring_tip = landmarks[16]['y']
        ring_pip = landmarks[14]['y']
        
        pinky_tip = landmarks[20]['y']
        pinky_pip = landmarks[18]['y']
        
        # Check if thumb is pointing upward and other fingers are curled
        thumb_extended_up = thumb_tip_y < thumb_ip_y and abs(thumb_tip_x - thumb_ip_x) < 0.1
        others_curled = (index_tip > index_pip and 
                         middle_tip > middle_pip and 
                         ring_tip > ring_pip and 
                         pinky_tip > pinky_pip)
        
        return thumb_extended_up and others_curled
    
    def _is_peace_sign(self, landmarks):
        # Check if index and middle fingers are extended but others are curled
        index_tip = landmarks[8]['y']
        index_pip = landmarks[6]['y']
        
        middle_tip = landmarks[12]['y']
        middle_pip = landmarks[10]['y']
        
        ring_tip = landmarks[16]['y']
        ring_pip = landmarks[14]['y']
        
        pinky_tip = landmarks[20]['y']
        pinky_pip = landmarks[18]['y']
        
        return (index_tip < index_pip and  # Index extended
                middle_tip < middle_pip and  # Middle extended
                ring_tip > ring_pip and  # Ring curled
                pinky_tip > pinky_pip)  # Pinky curled

def train_asl_model():
    """Use the pre-trained model from simple_local_train.py"""
    model_path = r"C:\Users\adity\OneDrive\Desktop\hand gesture\hand_gesture_model.pkl"
    
    if os.path.exists(model_path):
        print(f"Using pre-trained model from {model_path}")
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded with {len(model['X_train'])} samples")
            if 'y_train' in model:
                unique_letters = set(model['y_train'])
                print(f"Letters included: {unique_letters}")
            return model_path
        except Exception as e:
            print(f"Error loading model: {e}")
    
    print("Pre-trained model not found. Please run simple_local_train.py first.")
    return None

def main():
    # Path to the trained model
    model_path = r"C:\Users\adity\OneDrive\Desktop\hand gesture\hand_gesture_model.pkl"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, training new model...")
        model_path = train_asl_model()
    else:
        # Check if model has data
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            if len(model['X_train']) == 0:
                print("Model exists but has no data, training new model...")
                model_path = train_asl_model()
        except Exception as e:
            print(f"Error checking model: {e}")
            print("Training new model...")
            model_path = train_asl_model()
    
    # Initialize detector with model and improved settings
    detector = HandGestureDetector(
        static_image_mode=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_path=model_path
    )
    
    # Initialize webcam with higher resolution if available
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Disable face detection to improve performance
    use_face_detection = False
    
    # Training mode variables
    training_mode = False
    current_letter = None
    training_samples = []
    training_labels = []
    letters_to_train = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    letter_index = 0
    samples_per_letter = 20
    current_samples = 0
    countdown = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera. Check camera connection.")
                break
            
            # Flip the frame horizontally for a more intuitive mirror view
            frame = cv2.flip(frame, 1)
            
            try:
                # Process the image and get hand landmarks
                results = detector.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Handle training mode
                if training_mode:
                    if current_letter is None and letter_index < len(letters_to_train):
                        current_letter = letters_to_train[letter_index]
                        countdown = 50  # Give user time to prepare
                        current_samples = 0
                    
                    # Display training instructions
                    cv2.putText(
                        frame,
                        f"TRAINING MODE: Show '{current_letter}' sign",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                    
                    # Show countdown or sample count
                    if countdown > 0:
                        cv2.putText(
                            frame,
                            f"Get ready: {countdown//10}",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )
                        countdown -= 1
                    else:
                        cv2.putText(
                            frame,
                            f"Samples: {current_samples}/{samples_per_letter}",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )
                        
                        # Collect samples if hand is detected
                        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                            # Get the first hand
                            hand_landmarks = results.multi_hand_landmarks[0]
                            
                            # Extract landmark coordinates
                            landmarks = []
                            for landmark in hand_landmarks.landmark:
                                landmarks.append({
                                    'x': landmark.x,
                                    'y': landmark.y,
                                    'z': landmark.z
                                })
                            
                            # Normalize landmarks
                            normalized_landmarks = detector._normalize_landmarks(landmarks)
                            
                            # Add to training data
                            training_samples.append(normalized_landmarks)
                            training_labels.append(current_letter)
                            current_samples += 1
                            
                            # Wait a bit between samples
                            cv2.waitKey(100)
                    
                    # Move to next letter when enough samples collected
                    if current_samples >= samples_per_letter:
                        letter_index += 1
                        if letter_index < len(letters_to_train):
                            current_letter = letters_to_train[letter_index]
                            countdown = 50
                            current_samples = 0
                        else:
                            # Training complete, save model
                            model = {
                                'X_train': np.array(training_samples),
                                'y_train': np.array(training_labels),
                                'feature_weights': np.ones(len(training_samples[0]))
                            }
                            
                            with open(model_path, 'wb') as f:
                                pickle.dump(model, f)
                            
                            print(f"Model saved with {len(training_samples)} samples")
                            print(f"Letters trained: {set(training_labels)}")
                            
                            # Reload model
                            detector.asl_model = model
                            
                            # Exit training mode
                            training_mode = False
                    
                    # Draw hand landmarks if detected
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            detector.mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                detector.mp_hands.HAND_CONNECTIONS,
                                detector.mp_drawing_styles.get_default_hand_landmarks_style(),
                                detector.mp_drawing_styles.get_default_hand_connections_style()
                            )
                    
                    # Display the training frame
                    cv2.imshow('Hand Gesture Recognition', frame)
                    
                else:
                    # Normal mode (non-training)
                    # Detect hands and gestures
                    hands_data = detector.detect_gestures(frame)
                    
                    # Draw hand landmarks and gestures
                    output_frame = detector.draw_landmarks(frame, hands_data, results)
                    
                    # Detect faces (optional)
                    face_detections = []
                    if use_face_detection:
                        face_detections = detector.detect_faces(output_frame)
                        output_frame = detector.draw_faces(output_frame, face_detections)
                    
                    # Display ASL recognition status
                    if detector.asl_model is None:
                        cv2.putText(
                            output_frame,
                            "ASL Model Not Loaded",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )
                    else:
                        cv2.putText(
                            output_frame,
                            "ASL Recognition Active",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )
                    
                    # Add instructions
                    cv2.putText(
                        output_frame,
                        f"Press 'q' to quit, 'f' for face detection, 't' for training",
                        (10, output_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
                    
                    # Display result
                    cv2.imshow('Hand Gesture Recognition', output_frame)
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                # Display error frame
                cv2.putText(
                    frame,
                    f"Error: {str(e)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                cv2.imshow('Hand Gesture Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                use_face_detection = not use_face_detection
                print(f"Face detection: {'ON' if use_face_detection else 'OFF'}")
            elif key == ord('t'):
                if not training_mode:
                    training_mode = True
                    letter_index = 0
                    current_letter = None
                    training_samples = []
                    training_labels = []
                    print("Entering training mode...")
                else:
                    training_mode = False
                    print("Exiting training mode...")
    
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Release resources
        print("Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        print("Done")

if __name__ == "__main__":
    main()