import pickle
import numpy as np
from hand import HandGestureDetector

class EnhancedHandGestureDetector(HandGestureDetector):
    def __init__(self, model_path="hand_gesture_model.pkl", **kwargs):
        super().__init__(**kwargs)
        
        # Load trained model
        try:
            with open(model_path, 'rb') as f:
                self.trained_model = pickle.load(f)
            self.model_loaded = True
            print(f"Loaded trained gesture model from {model_path}")
        except:
            self.model_loaded = False
            print(f"Could not load model from {model_path}")
    
    def detect_gestures(self, image):
        # Get basic hand data using parent method
        hands_data = super().detect_gestures(image)
        
        # If model is loaded, enhance with trained gestures
        if self.model_loaded and hands_data:
            for hand_data in hands_data:
                # Extract landmarks in the format expected by the model
                landmarks = hand_data['landmarks']
                features = []
                for landmark in landmarks:
                    features.extend([landmark['x'], landmark['y'], landmark['z']])
                
                # Predict gesture using trained model
                features = np.array(features).reshape(1, -1)
                predicted_gesture = self.trained_model.predict(features)[0]
                
                # Add predicted gesture to the list
                hand_data['gestures'].append(f"trained_{predicted_gesture}")
        
        return hands_data

# Example usage
if __name__ == "__main__":
    import cv2
    
    # Initialize enhanced detector with trained model
    detector = EnhancedHandGestureDetector(
        model_path="hand_gesture_model.pkl",
        min_detection_confidence=0.7
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)
        
        # Process the image and get hand landmarks
        results = detector.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Detect hands and gestures (including trained gestures)
        hands_data = detector.detect_gestures(frame)
        
        # Draw hand landmarks and gestures
        output_frame = detector.draw_landmarks(frame, hands_data, results)
        
        # Detect faces
        face_detections = detector.detect_faces(output_frame)
        
        # Draw face detections
        output_frame = detector.draw_faces(output_frame, face_detections)
        
        # Display result
        cv2.imshow('Enhanced Hand and Face Detection', output_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()