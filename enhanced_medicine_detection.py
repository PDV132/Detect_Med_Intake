"""
Enhanced Medicine Detection with Strict Validation
Validates three critical conditions:
1. One hand holding medicine strip/bottle
2. Other hand with fingers near mouth
3. Head tilted back for taking medicine
"""

import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class StrictMedicineDetector:
    def __init__(self):
        # Initialize MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize models with higher confidence for strict detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Must detect both hands
            min_detection_confidence=0.8,  # Higher confidence
            min_tracking_confidence=0.7
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.8
        )
        
        # Strict detection parameters
        self.hand_mouth_distance_threshold = 0.12  # Stricter threshold
        self.head_tilt_threshold = -0.08  # More pronounced head tilt required
        self.medicine_object_confidence_threshold = 0.7  # Higher confidence for object detection
        
        # Validation requirements
        self.min_consecutive_frames = 10  # Must be detected for multiple frames
        self.validation_buffer = []
        
    def detect_medicine_strip_bottle(self, image: np.ndarray, hand_landmarks, hand_label: str) -> Dict[str, Any]:
        """
        Enhanced medicine object detection with stricter validation
        """
        h, w, _ = image.shape
        
        # Get hand bounding box
        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Expand region to look for medicine objects
        padding = 60
        roi_x1 = max(0, x_min - padding)
        roi_y1 = max(0, y_min - padding)
        roi_x2 = min(w, x_max + padding)
        roi_y2 = min(h, y_max + padding)
        
        hand_roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if hand_roi.size == 0:
            return {"medicine_object_detected": False, "object_type": None, "confidence": 0.0}
        
        # Enhanced object detection with multiple validation methods
        medicine_detected = self._analyze_medicine_object_enhanced(hand_roi, hand_label)
        
        return medicine_detected
    
    def _analyze_medicine_object_enhanced(self, roi: np.ndarray, hand_label: str) -> Dict[str, Any]:
        """
        Enhanced medicine object analysis with multiple detection methods
        """
        # Convert to different color spaces for better analysis
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Color-based detection for medicine strips (metallic/silver)
        strip_confidence = self._detect_medicine_strip(hsv, gray)
        
        # Method 2: Shape and texture analysis for bottles
        bottle_confidence = self._detect_medicine_bottle(hsv, gray)
        
        # Method 3: Edge detection for rectangular medicine packages
        package_confidence = self._detect_medicine_package(gray)
        
        # Determine best detection
        max_confidence = max(strip_confidence, bottle_confidence, package_confidence)
        
        if max_confidence > self.medicine_object_confidence_threshold:
            if strip_confidence == max_confidence:
                object_type = "medicine_strip"
            elif bottle_confidence == max_confidence:
                object_type = "medicine_bottle"
            else:
                object_type = "medicine_package"
                
            return {
                "medicine_object_detected": True,
                "object_type": object_type,
                "confidence": max_confidence,
                "hand_holding": hand_label,
                "detection_method": "enhanced_multi_method"
            }
        
        return {
            "medicine_object_detected": False,
            "object_type": None,
            "confidence": max_confidence,
            "hand_holding": hand_label
        }
    
    def _detect_medicine_strip(self, hsv: np.ndarray, gray: np.ndarray) -> float:
        """Detect medicine strip (blister pack) with metallic backing"""
        # Silver/metallic color detection
        silver_lower = np.array([0, 0, 180])
        silver_upper = np.array([255, 40, 255])
        silver_mask = cv2.inRange(hsv, silver_lower, silver_upper)
        
        # Aluminum foil detection (common in medicine strips)
        aluminum_lower = np.array([0, 0, 200])
        aluminum_upper = np.array([255, 25, 255])
        aluminum_mask = cv2.inRange(hsv, aluminum_lower, aluminum_upper)
        
        # Combine masks
        metallic_mask = cv2.bitwise_or(silver_mask, aluminum_mask)
        
        # Calculate confidence based on metallic area and shape
        metallic_area = cv2.countNonZero(metallic_mask)
        total_area = hsv.shape[0] * hsv.shape[1]
        area_ratio = metallic_area / total_area
        
        # Look for rectangular shapes (medicine strips are typically rectangular)
        contours, _ = cv2.findContours(metallic_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_confidence = 0.0
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Minimum area
                # Check if contour is roughly rectangular
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) >= 4:  # Roughly rectangular
                    shape_confidence = max(shape_confidence, 0.8)
        
        # Combine area and shape confidence
        confidence = min(1.0, (area_ratio * 3) + (shape_confidence * 0.5))
        return confidence
    
    def _detect_medicine_bottle(self, hsv: np.ndarray, gray: np.ndarray) -> float:
        """Detect medicine bottle (plastic/glass container)"""
        # White/transparent bottle detection
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([255, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # Orange/amber bottle detection (common for prescription bottles)
        orange_lower = np.array([8, 100, 100])
        orange_upper = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        
        # Brown bottle detection
        brown_lower = np.array([5, 50, 50])
        brown_upper = np.array([15, 255, 200])
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        
        # Combine masks
        bottle_mask = cv2.bitwise_or(cv2.bitwise_or(white_mask, orange_mask), brown_mask)
        
        # Calculate confidence
        bottle_area = cv2.countNonZero(bottle_mask)
        total_area = hsv.shape[0] * hsv.shape[1]
        area_ratio = bottle_area / total_area
        
        # Look for cylindrical/circular shapes
        contours, _ = cv2.findContours(bottle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_confidence = 0.0
        
        for contour in contours:
            if cv2.contourArea(contour) > 200:  # Minimum area for bottle
                # Check circularity (bottles appear circular from certain angles)
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:  # Reasonably circular
                        shape_confidence = max(shape_confidence, circularity)
        
        confidence = min(1.0, (area_ratio * 2.5) + (shape_confidence * 0.7))
        return confidence
    
    def _detect_medicine_package(self, gray: np.ndarray) -> float:
        """Detect medicine package using edge detection"""
        # Edge detection for rectangular packages
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        package_confidence = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area for medicine package
                # Check if it's rectangular
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:  # Rectangle
                    # Check aspect ratio (medicine packages are typically rectangular)
                    rect = cv2.boundingRect(contour)
                    aspect_ratio = rect[2] / rect[3]  # width/height
                    
                    if 0.5 <= aspect_ratio <= 3.0:  # Reasonable aspect ratio
                        package_confidence = max(package_confidence, 0.8)
        
        return package_confidence
    
    def validate_medicine_intake_conditions(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Strict validation of all three medicine intake conditions:
        1. One hand holding medicine strip/bottle
        2. Other hand with fingers near mouth
        3. Head tilted back
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process all components
        hand_results = self.hands.process(rgb_image)
        pose_results = self.pose.process(rgb_image)
        face_results = self.face_detection.process(rgb_image)
        
        validation_result = {
            "condition_1_medicine_in_hand": False,
            "condition_2_hand_near_mouth": False,
            "condition_3_head_tilted_back": False,
            "all_conditions_met": False,
            "medicine_object_details": None,
            "hand_mouth_details": None,
            "head_tilt_details": None,
            "confidence_scores": {},
            "timestamp": time.time(),
            "validation_strict": True
        }
        
        # CONDITION 1 & 2: Check hands (must detect exactly 2 hands)
        if not (hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) == 2):
            validation_result["failure_reason"] = "Must detect exactly 2 hands - one holding medicine, one near mouth"
            return validation_result
        
        # Get hand classifications (Left/Right)
        hand_classifications = []
        if hand_results.multi_handedness:
            for classification in hand_results.multi_handedness:
                hand_classifications.append(classification.classification[0].label)
        
        # CONDITION 3: Check head tilt first
        head_tilt_valid = False
        if pose_results.pose_landmarks:
            nose = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_ear = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
            
            # Calculate head tilt (nose should be higher than ears when head tilted back)
            ear_center_y = (left_ear.y + right_ear.y) / 2
            head_tilt_ratio = nose.y - ear_center_y
            
            if head_tilt_ratio < self.head_tilt_threshold:
                head_tilt_valid = True
                validation_result["condition_3_head_tilted_back"] = True
                validation_result["head_tilt_details"] = {
                    "tilt_ratio": head_tilt_ratio,
                    "threshold": self.head_tilt_threshold,
                    "tilt_angle_estimate": abs(head_tilt_ratio) * 90  # Rough angle estimate
                }
                validation_result["confidence_scores"]["head_tilt"] = min(1.0, abs(head_tilt_ratio) / abs(self.head_tilt_threshold))
        
        if not head_tilt_valid:
            validation_result["failure_reason"] = "Head not tilted back sufficiently for medicine intake"
            return validation_result
        
        # Get face reference for mouth position
        if not face_results.detections:
            validation_result["failure_reason"] = "Face not detected for mouth reference"
            return validation_result
        
        face_detection = face_results.detections[0]
        face_bbox = face_detection.location_data.relative_bounding_box
        face_center_x = face_bbox.xmin + face_bbox.width / 2
        face_center_y = face_bbox.ymin + face_bbox.height / 2
        mouth_area_x = face_center_x
        mouth_area_y = face_center_y + face_bbox.height * 0.3  # Lower part of face (mouth area)
        
        # Analyze both hands for conditions 1 and 2
        medicine_hand_found = False
        mouth_hand_found = False
        
        for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            hand_label = hand_classifications[hand_idx] if hand_idx < len(hand_classifications) else f"Hand_{hand_idx}"
            
            # Check if this hand is holding medicine (CONDITION 1)
            medicine_detection = self.detect_medicine_strip_bottle(image, hand_landmarks, hand_label)
            
            if medicine_detection["medicine_object_detected"]:
                medicine_hand_found = True
                validation_result["condition_1_medicine_in_hand"] = True
                validation_result["medicine_object_details"] = medicine_detection
                validation_result["confidence_scores"]["medicine_object"] = medicine_detection["confidence"]
                continue  # This hand is holding medicine, check the other hand for mouth proximity
            
            # Check if this hand is near mouth (CONDITION 2)
            # Use multiple finger tips for better accuracy
            finger_tips = [
                hand_landmarks.landmark[4],   # Thumb tip
                hand_landmarks.landmark[8],   # Index finger tip
                hand_landmarks.landmark[12],  # Middle finger tip
                hand_landmarks.landmark[16],  # Ring finger tip
                hand_landmarks.landmark[20]   # Pinky tip
            ]
            
            # Calculate minimum distance from any finger tip to mouth
            min_distance = float('inf')
            closest_finger = None
            
            for i, finger_tip in enumerate(finger_tips):
                distance = np.sqrt(
                    (finger_tip.x - mouth_area_x) ** 2 + 
                    (finger_tip.y - mouth_area_y) ** 2
                )
                if distance < min_distance:
                    min_distance = distance
                    closest_finger = i
            
            if min_distance < self.hand_mouth_distance_threshold:
                mouth_hand_found = True
                validation_result["condition_2_hand_near_mouth"] = True
                validation_result["hand_mouth_details"] = {
                    "distance": min_distance,
                    "threshold": self.hand_mouth_distance_threshold,
                    "closest_finger": ["thumb", "index", "middle", "ring", "pinky"][closest_finger],
                    "hand_label": hand_label
                }
                validation_result["confidence_scores"]["hand_mouth_distance"] = 1 - (min_distance / self.hand_mouth_distance_threshold)
        
        # Final validation: All three conditions must be met
        if medicine_hand_found and mouth_hand_found and head_tilt_valid:
            validation_result["all_conditions_met"] = True
            validation_result["validation_status"] = "VALID_MEDICINE_INTAKE"
            
            # Calculate overall confidence
            confidence_values = list(validation_result["confidence_scores"].values())
            validation_result["overall_confidence"] = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        else:
            # Determine specific failure reason
            missing_conditions = []
            if not medicine_hand_found:
                missing_conditions.append("medicine object in hand")
            if not mouth_hand_found:
                missing_conditions.append("hand near mouth")
            if not head_tilt_valid:
                missing_conditions.append("head tilted back")
            
            validation_result["failure_reason"] = f"Missing conditions: {', '.join(missing_conditions)}"
            validation_result["validation_status"] = "INVALID_MEDICINE_INTAKE"
        
        return validation_result
    
    def process_video_with_strict_validation(self, video_path: str) -> Dict[str, Any]:
        """
        Process video with strict validation of medicine intake conditions
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "error", "error": "Could not open video file"}
        
        valid_detections = []
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        consecutive_valid_frames = 0
        
        logger.info("Starting strict medicine intake validation...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp_sec = frame_count / fps
            
            # Validate every 3 frames for efficiency
            if frame_count % 3 == 0:
                validation = self.validate_medicine_intake_conditions(frame)
                
                if validation["all_conditions_met"]:
                    consecutive_valid_frames += 1
                    
                    event = {
                        "timestamp_sec": timestamp_sec,
                        "frame_number": frame_count,
                        "validation_details": validation,
                        "consecutive_frame": consecutive_valid_frames
                    }
                    valid_detections.append(event)
                    
                    logger.info(f"Valid medicine intake detected at {timestamp_sec:.1f}s (frame {consecutive_valid_frames})")
                else:
                    consecutive_valid_frames = 0
        
        cap.release()
        
        # Determine final result based on strict criteria
        if len(valid_detections) >= self.min_consecutive_frames:
            return {
                "status": "taken",
                "validation_method": "strict_three_condition_check",
                "events": valid_detections,
                "total_valid_frames": len(valid_detections),
                "consecutive_valid_frames": max([event["consecutive_frame"] for event in valid_detections]),
                "total_frames": frame_count,
                "duration_sec": frame_count / fps,
                "conditions_validated": [
                    "Medicine object (strip/bottle) in one hand",
                    "Other hand fingers near mouth",
                    "Head tilted back for swallowing"
                ]
            }
        else:
            # Provide detailed feedback on what was missing
            failure_reasons = []
            if valid_detections:
                last_validation = valid_detections[-1]["validation_details"]
                if not last_validation["condition_1_medicine_in_hand"]:
                    failure_reasons.append("No medicine strip or bottle detected in hand")
                if not last_validation["condition_2_hand_near_mouth"]:
                    failure_reasons.append("Hand not positioned near mouth")
                if not last_validation["condition_3_head_tilted_back"]:
                    failure_reasons.append("Head not tilted back for swallowing")
            else:
                failure_reasons.append("No valid medicine intake gestures detected")
            
            return {
                "status": "missed",
                "validation_method": "strict_three_condition_check",
                "reason": "Strict validation failed",
                "failure_details": failure_reasons,
                "partial_detections": len(valid_detections),
                "required_consecutive_frames": self.min_consecutive_frames,
                "total_frames": frame_count,
                "duration_sec": frame_count / fps,
                "conditions_required": [
                    "Medicine object (strip/bottle) in one hand",
                    "Other hand fingers near mouth", 
                    "Head tilted back for swallowing"
                ]
            }
    
    def release(self):
        """Clean up resources"""
        if hasattr(self, 'hands'):
            self.hands.close()
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'face_detection'):
            self.face_detection.close()


# Main function for testing
if __name__ == "__main__":
    import sys
    
    detector = StrictMedicineDetector()
    
    if len(sys.argv) > 1:
        result = detector.process_video_with_strict_validation(sys.argv[1])
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python enhanced_medicine_detection.py <video_file>")
        print("\nThis detector validates three strict conditions:")
        print("1. One hand holding medicine strip or bottle")
        print("2. Other hand with fingers near mouth")
        print("3. Head tilted back for taking medicine")
    
    detector.release()
