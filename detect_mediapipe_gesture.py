# detect_mediapipe_gesture.py
import cv2
import mediapipe as mp
import numpy as np

class GestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_gesture(self, image):
        """
        Detect hand gestures in an image
        Returns: gesture_name, landmarks, image_with_annotations
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(rgb_image)
        
        gesture_name = "No gesture"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Simple gesture detection logic
                gesture_name = self._classify_gesture(hand_landmarks)
        
        return gesture_name, results.multi_hand_landmarks, image

    def _classify_gesture(self, landmarks):
        """
        Simple gesture classification
        You can expand this with more sophisticated logic
        """
        # Get landmark positions
        landmarks_array = []
        for lm in landmarks.landmark:
            landmarks_array.append([lm.x, lm.y])
        
        landmarks_array = np.array(landmarks_array)
        
        # Simple gesture detection (you can improve this)
        # For now, just return "Hand detected"
        return "Hand detected"

    def release(self):
        """Clean up resources"""
        self.hands.close()

# Convenience functions for easy import
def init_gesture_detector():
    """Initialize and return a gesture detector"""
    return GestureDetector()

def detect_gesture_in_frame(detector, frame):
    """Detect gesture in a single frame"""
    return detector.detect_gesture(frame)