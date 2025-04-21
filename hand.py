import os
import cv2
import mediapipe as mp
import numpy np
import pickle
import time
import re
from gtts import gTTS
from playsound import playsound


def speak(text, lang='en'):
    """Convert text to speech using gTTS and play using the laptop's speaker."""
    try:
        tts = gTTS(text=text, lang=lang)
        filename = "temp_audio.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        print(f"Speech error: {e}")


class HandGestureDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_path="hand_gesture_model.pkl"):
        # Initialize MediaPipe Hands solution with improved parameters
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize landmark smoothing
        self.smoothing_factor = 0.5
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

    def recognize_asl(self, landmarks):
        if self.asl_model is None:
            return None
        try:
            normalized_landmarks = self._normalize_landmarks(landmarks)
            X_train = np.array(self.asl_model['X_train'])
            y_train = np.array(self.asl_model['y_train'])
            if len(normalized_landmarks) != X_train.shape[1]:
                if len(normalized_landmarks) > X_train.shape[1]:
                    normalized_landmarks = normalized_landmarks[:X_train.shape[1]]
                else:
                    normalized_landmarks = np.pad(normalized_landmarks, 
                                                  (0, X_train.shape[1] - len(normalized_landmarks)),
                                                  'constant')
            feature_weights = self.asl_model.get('feature_weights', np.ones(len(normalized_landmarks)))
            if len(X_train) == 0 or len(y_train) == 0:
                return "No Data"
            weighted_landmarks = np.array(normalized_landmarks) * feature_weights
            distances = np.sqrt(np.sum((X_train - weighted_landmarks)**2, axis=1))
            k = min(7, len(y_train))
            k_nearest_indices = np.argsort(distances)[:k]
            label_weights = {}
            for label, dist in zip(y_train[k_nearest_indices], distances[k_nearest_indices]):
                weight = 1.0 / (dist**2 + 1e-5)
                label_weights[label] = label_weights.get(label, 0) + weight
            if not label_weights:
                return None
            predicted_letter = max(label_weights, key=label_weights.get)
            total_weight = sum(label_weights.values())
            confidence = label_weights[predicted_letter] / total_weight if total_weight > 0 else 0
            if confidence > 0.6:
                return predicted_letter
            else:
                return None
        except Exception as e:
            print(f"Error in ASL recognition: {e}")
            return None

    def _normalize_landmarks(self, landmarks):
        coords = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        centroid = np.mean(coords, axis=0)
        centered = coords - centroid
        max_distance = np.max(np.linalg.norm(centered, axis=1))
        normalized = centered / max_distance if max_distance > 0 else centered
        normalized_flat = normalized.flatten()
        angles = self._calculate_finger_angles(coords)
        combined_features = np.concatenate([normalized_flat, angles])
        return combined_features

    def _calculate_finger_angles(self, coords):
        angles = []
        fingers = [
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [0, 9, 10, 11, 12],
            [0, 13, 14, 15, 16],
            [0, 17, 18, 19, 20]
        ]
        for finger in fingers:
            for i in range(1, len(finger)-1):
                p1 = coords[finger[i-1]]
                p2 = coords[finger[i]]
                p3 = coords[finger[i+1]]
                v1 = p1 - p2
                v2 = p3 - p2
                dot = np.dot(v1, v2)
                norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                angle = np.arccos(np.clip(dot / norm, -1.0, 1.0)) if norm >= 1e-6 else 0
                angles.append(angle)
        return np.array(angles)

    def detect_gestures(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        hands_data = []
        if results.multi_hand_landmarks:
            if len(self.previous_landmarks) != len(results.multi_hand_landmarks):
                self.previous_landmarks = [None] * len(results.multi_hand_landmarks)
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = "Right" if results.multi_handedness[hand_idx].classification[0].label == "Right" else "Left"
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append({'x': landmark.x, 'y': landmark.y, 'z': landmark.z})
                if self.previous_landmarks[hand_idx] is not None:
                    landmarks = self._smooth_landmarks(landmarks, self.previous_landmarks[hand_idx])
                self.previous_landmarks[hand_idx] = landmarks
                detected_gestures = []
                for gesture_name, gesture_func in self.gestures.items():
                    if gesture_func(landmarks):
                        detected_gestures.append(gesture_name)
                asl_letter = self.recognize_asl(landmarks)
                hands_data.append({
                    'hand_label': hand_label,
                    'landmarks': landmarks,
                    'gestures': detected_gestures,
                    'asl_letter': asl_letter
                })
        else:
            self.previous_landmarks = []
        return hands_data

    def _smooth_landmarks(self, current_landmarks, previous_landmarks):
        smoothed_landmarks = []
        for i, landmark in enumerate(current_landmarks):
            prev = previous_landmarks[i]
            smoothed = {
                'x': prev['x'] * self.smoothing_factor + landmark['x'] * (1 - self.smoothing_factor),
                'y': prev['y'] * self.smoothing_factor + landmark['y'] * (1 - self.smoothing_factor),
                'z': prev['z'] * self.smoothing_factor + landmark['z'] * (1 - self.smoothing_factor)
            }
            smoothed_landmarks.append(smoothed)
        return smoothed_landmarks

    def draw_landmarks(self, image, hands_data, results):
        output_image = image.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    output_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=5, circle_radius=5),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=3)
                )
        for idx, hand_data in enumerate(hands_data):
            wrist = hand_data['landmarks'][0]
            wrist_pos = (int(wrist['x'] * output_image.shape[1]), int(wrist['y'] * output_image.shape[0]))
            text_bg_color = (0, 0, 0)
            text_bg_alpha = 0.5
            text_padding = 10
            text_y_offset = 30
            if hand_data.get('asl_letter'):
                asl_text = f"ASL: {hand_data['asl_letter']}"
                text_size = cv2.getTextSize(asl_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                asl_text_pos = (wrist_pos[0], wrist_pos[1] - text_y_offset)
                bg_rect_pt1 = (asl_text_pos[0] - text_padding, asl_text_pos[1] - text_size[1] - text_padding)
                bg_rect_pt2 = (asl_text_pos[0] + text_size[0] + text_padding, asl_text_pos[1] + text_padding)
                overlay = output_image.copy()
                cv2.rectangle(overlay, bg_rect_pt1, bg_rect_pt2, text_bg_color, -1)
                output_image = cv2.addWeighted(overlay, text_bg_alpha, output_image, 1 - text_bg_alpha, 0)
                cv2.putText(output_image, asl_text, asl_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                text_y_offset += 40
            if hand_data['gestures']:
                for i, gesture in enumerate(hand_data['gestures']):
                    gesture_text = f"{hand_data['hand_label']}: {gesture}"
                    text_size = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    text_pos = (wrist_pos[0], wrist_pos[1] - text_y_offset + (i * 35))
                    bg_rect_pt1 = (text_pos[0] - text_padding, text_pos[1] - text_size[1] - text_padding)
                    bg_rect_pt2 = (text_pos[0] + text_size[0] + text_padding, text_pos[1] + text_padding)
                    overlay = output_image.copy()
                    cv2.rectangle(overlay, bg_rect_pt1, bg_rect_pt2, text_bg_color, -1)
                    output_image = cv2.addWeighted(overlay, text_bg_alpha, output_image, 1 - text_bg_alpha, 0)
                    cv2.putText(output_image, gesture_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return output_image

    def detect_faces(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        face_detections = []
        if results.detections:
            for detection in results.detections:
                bounding_box = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x = int(bounding_box.xmin * w)
                y = int(bounding_box.ymin * h)
                width = int(bounding_box.width * w)
                height = int(bounding_box.height * h)
                face_detections.append({'box': (x, y, width, height), 'confidence': detection.score[0]})
        return face_detections

    def draw_faces(self, image, face_detections):
        output_image = image.copy()
        for detection in face_detections:
            x, y, width, height = detection['box']
            cv2.rectangle(output_image, (x, y), (x + width, y + height), (0, 0, 255), 2)
            text = f"Face: {detection['confidence']:.2f}"
            cv2.putText(output_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return output_image

    def _is_open_palm(self, landmarks):
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
        return (thumb_tip < thumb_ip and 
                index_tip < index_pip and 
                middle_tip < middle_pip and 
                ring_tip < ring_pip and 
                pinky_tip < pinky_pip)

    def _is_closed_fist(self, landmarks):
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
        return (thumb_tip > thumb_ip and 
                index_tip > index_pip and 
                middle_tip > middle_pip and 
                ring_tip > ring_pip and 
                pinky_tip > pinky_pip)

    def _is_pointing(self, landmarks):
        index_tip = landmarks[8]['y']
        index_pip = landmarks[6]['y']
        middle_tip = landmarks[12]['y']
        middle_pip = landmarks[10]['y']
        ring_tip = landmarks[16]['y']
        ring_pip = landmarks[14]['y']
        pinky_tip = landmarks[20]['y']
        pinky_pip = landmarks[18]['y']
        return (index_tip < index_pip and 
                middle_tip > middle_pip and 
                ring_tip > ring_pip and 
                pinky_tip > pinky_pip)

    def _is_thumbs_up(self, landmarks):
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
        thumb_extended_up = thumb_tip_y < thumb_ip_y and abs(thumb_tip_x - thumb_ip_x) < 0.1
        others_curled = (index_tip > index_pip and 
                         middle_tip > middle_pip and 
                         ring_tip > ring_pip and 
                         pinky_tip > pinky_pip)
        return thumb_extended_up and others_curled

    def _is_peace_sign(self, landmarks):
        index_tip = landmarks[8]['y']
        index_pip = landmarks[6]['y']
        middle_tip = landmarks[12]['y']
        middle_pip = landmarks[10]['y']
        ring_tip = landmarks[16]['y']
        ring_pip = landmarks[14]['y']
        pinky_tip = landmarks[20]['y']
        pinky_pip = landmarks[18]['y']
        return (index_tip < index_pip and 
                middle_tip < middle_pip and 
                ring_tip > ring_pip and 
                pinky_tip > pinky_pip)

    def _is_flat_hand(self, landmarks):
        index_tip_y = landmarks[8]['y']
        middle_tip_y = landmarks[12]['y']
        ring_tip_y = landmarks[16]['y']
        pinky_tip_y = landmarks[20]['y']
        fingertips_aligned = (abs(index_tip_y - middle_tip_y) < 0.05 and
                              abs(middle_tip_y - ring_tip_y) < 0.05 and
                              abs(ring_tip_y - pinky_tip_y) < 0.05)
        index_extended = landmarks[8]['y'] < landmarks[6]['y']
        middle_extended = landmarks[12]['y'] < landmarks[10]['y']
        ring_extended = landmarks[16]['y'] < landmarks[14]['y']
        pinky_extended = landmarks[20]['y'] < landmarks[18]['y']
        return fingertips_aligned and index_extended and middle_extended and ring_extended and pinky_extended

    def _is_swipe_left(self, landmarks, prev_landmarks):
        if prev_landmarks is None:
            return False
        current_wrist_x = landmarks[0]['x']
        prev_wrist_x = prev_landmarks[0]['x']
        return (prev_wrist_x - current_wrist_x) > 0.15

def train_asl_model():
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
    model_path = r"C:\Users\adity\OneDrive\Desktop\hand gesture\hand_gesture_model.pkl"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, training new model...")
        model_path = train_asl_model()
    else:
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

    detector = HandGestureDetector(
        static_image_mode=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_path=model_path
    )
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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

    # Sentence building mode variables
    sentence_mode = False
    current_sentence = ""
    current_word = ""
    last_letter = None
    letter_stability_count = 0
    required_stability = 10  # frames needed for stable detection
    last_letter_time = time.time()
    letter_timeout = 1.0
    word_timeout = 3.0
    last_word_time = time.time()
    prev_hand_landmarks = None

    # For normal mode voice output (speak recognized letters after a 15-sec stability period)
    last_normal_letter = None
    stable_letter = None
    normal_letter_start_time = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera. Check camera connection.")
                break
            frame = cv2.flip(frame, 1)
            results = detector.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # ---------------- Training Mode ----------------
            if training_mode:
                if current_letter is None and letter_index < len(letters_to_train):
                    current_letter = letters_to_train[letter_index]
                    countdown = 50
                    current_samples = 0
                cv2.putText(frame, f"TRAINING MODE: Show '{current_letter}' sign", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if countdown > 0:
                    cv2.putText(frame, f"Get ready: {countdown//10}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    countdown -= 1
                else:
                    cv2.putText(frame, f"Samples: {current_samples}/{samples_per_letter}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.append({'x': landmark.x, 'y': landmark.y, 'z': landmark.z})
                        normalized_landmarks = detector._normalize_landmarks(landmarks)
                        training_samples.append(normalized_landmarks)
                        training_labels.append(current_letter)
                        current_samples += 1
                        cv2.waitKey(100)
                if current_samples >= samples_per_letter:
                    letter_index += 1
                    if letter_index < len(letters_to_train):
                        current_letter = letters_to_train[letter_index]
                        countdown = 50
                        current_samples = 0
                    else:
                        model = {
                            'X_train': np.array(training_samples),
                            'y_train': np.array(training_labels),
                            'feature_weights': np.ones(len(training_samples[0]))
                        }
                        with open(model_path, 'wb') as f:
                            pickle.dump(model, f)
                        print(f"Model saved with {len(training_samples)} samples")
                        print(f"Letters trained: {set(training_labels)}")
                        detector.asl_model = model
                        training_mode = False
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        detector.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            detector.mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=detector.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=5, circle_radius=5),
                            connection_drawing_spec=detector.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=3)
                        )
                cv2.imshow('Hand Gesture Recognition', frame)
            
            # ---------------- Sentence Mode ----------------
            elif sentence_mode:
                hands_data = detector.detect_gestures(frame)
                output_frame = detector.draw_landmarks(frame, hands_data, results)
                cv2.putText(output_frame, "SENTENCE MODE", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(output_frame, f"Sentence: {current_sentence + current_word}", (10, output_frame.shape[0]-50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(output_frame, f"Current word: {current_word}", (10, output_frame.shape[0]-80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if hands_data and len(hands_data) > 0:
                    hand_data = hands_data[0]
                    landmarks = hand_data['landmarks']
                    # Finalize a word if a flat hand gesture is detected
                    if detector._is_flat_hand(landmarks):
                        if current_word:
                            current_sentence += current_word + " "
                            print(f"Word added. Sentence: {current_sentence}")
                            speak(current_word)
                            current_word = ""
                            last_word_time = time.time()
                    elif detector._is_closed_fist(landmarks):
                        if current_word:
                            current_word = current_word[:-1]
                            print(f"Character deleted. Word: {current_word}")
                        elif current_sentence:
                            current_sentence = re.sub(r'\s*\S+\s*$', '', current_sentence)
                            print(f"Word deleted. Sentence: {current_sentence}")
                    elif detector._is_swipe_left(landmarks, prev_hand_landmarks):
                        current_word = ""
                        current_sentence = ""
                        print("Sentence cleared")
                    elif hand_data.get('asl_letter'):
                        detected_letter = hand_data['asl_letter']
                        current_time = time.time()
                        if (detected_letter != last_letter or current_time - last_letter_time > letter_timeout):
                            if detected_letter == last_letter:
                                letter_stability_count += 1
                            else:
                                letter_stability_count = 1
                                last_letter = detected_letter
                            if letter_stability_count >= required_stability:
                                current_word += detected_letter
                                letter_stability_count = 0
                                last_letter_time = current_time
                                print(f"Letter added: {detected_letter}. Word: {current_word}")
                    prev_hand_landmarks = landmarks
                current_time = time.time()
                if current_word and current_time - last_word_time > word_timeout:
                    print(f"Auto-completing word: {current_word}")
                    current_sentence += current_word
                    speak(current_word)
                    current_word = ""
                    last_word_time = current_time
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('x') and current_sentence:
                    # Speak the entire sentence without pauses
                    sentence_to_speak = current_sentence + current_word
                    sentence_to_speak = sentence_to_speak.strip()  # Remove trailing spaces
                    if sentence_to_speak:
                        print(f"Speaking full sentence: {sentence_to_speak}")
                        speak(sentence_to_speak)

                cv2.imshow('Hand Gesture Recognition', output_frame)

            # ---------------- Normal Mode ----------------
            else:
                hands_data = detector.detect_gestures(frame)
                output_frame = detector.draw_landmarks(frame, hands_data, results)
                face_detections = []
                if use_face_detection:
                    face_detections = detector.detect_faces(output_frame)
                    output_frame = detector.draw_faces(output_frame, face_detections)
                if detector.asl_model is None:
                    cv2.putText(output_frame, "ASL Model Not Loaded", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(output_frame, "ASL Recognition Active", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(output_frame, "Press 'q' to quit, 'f' for face detection, 't' for training, 's' for sentence mode, 'x' to speak sentence",
                            (10, output_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # In normal mode, require 15 seconds of stability in letter detection before speaking the letter.
                if hands_data and len(hands_data) > 0:
                    current_letter_detected = hands_data[0].get('asl_letter', None)
                    if current_letter_detected:
                        if current_letter_detected != stable_letter:
                            stable_letter = current_letter_detected
                            normal_letter_start_time = time.time()
                        else:
                            if normal_letter_start_time is not None and (time.time() - normal_letter_start_time) >= 15:
                                if current_letter_detected != last_normal_letter:
                                    print(f"Speaking new letter after delay: {current_letter_detected}")
                                    speak(current_letter_detected)
                                    last_normal_letter = current_letter_detected
                cv2.imshow('Hand Gesture Recognition', output_frame)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                use_face_detection = not use_face_detection
                print(f"Face detection: {'ON' if use_face_detection else 'OFF'}")
            elif key == ord('t'):
                if not training_mode:
                    training_mode = True
                    sentence_mode = False
                    letter_index = 0
                    current_letter = None
                    training_samples = []
                    training_labels = []
                    print("Entering training mode...")
                else:
                    training_mode = False
                    print("Exiting training mode...")
            elif key == ord('s'):
                sentence_mode = not sentence_mode
                if sentence_mode:
                    training_mode = False
                    current_sentence = ""
                    current_word = ""
                    last_letter = None
                    letter_stability_count = 0
                    last_letter_time = time.time()
                    last_word_time = time.time()
                    prev_hand_landmarks = None
                    print("Entering sentence building mode...")
                else:
                    print("Exiting sentence building mode...")

    except KeyboardInterrupt:
        print("Program interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

speak("Hello, this is a test.")