# enhanced_detect_mediapipe_gesture.py
import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
import time
from datetime import datetime, timedelta
import threading
import schedule
import json

logger = logging.getLogger(__name__)

class EnhancedMedicineDetector:
    def __init__(self):
        # Initialize MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize models
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.7
        )
        
        # Detection parameters
        self.medicine_detection_threshold = 0.6
        self.gesture_sequence_frames = 15
        self.hand_mouth_distance_threshold = 0.15
        
        # Live monitoring
        self.is_monitoring = False
        self.monitoring_thread = None
        self.scheduled_times = []  # List of daily monitoring times
        
    def detect_medicine_strip_bottle(self, image: np.ndarray, hand_landmarks) -> Dict[str, Any]:
        """
        Detect if person is holding a medicine strip or bottle
        Uses object detection and hand position analysis
        """
        h, w, _ = image.shape
        
        # Get hand bounding box
        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Expand region to look for medicine objects
        padding = 50
        roi_x1 = max(0, x_min - padding)
        roi_y1 = max(0, y_min - padding)
        roi_x2 = min(w, x_max + padding)
        roi_y2 = min(h, y_max + padding)
        
        hand_roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if hand_roi.size == 0:
            return {"medicine_object_detected": False, "object_type": None}
        
        # Simple object detection based on color and shape analysis
        medicine_detected = self._analyze_medicine_object(hand_roi)
        
        return medicine_detected
    
    def _analyze_medicine_object(self, roi: np.ndarray) -> Dict[str, Any]:
        """
        Analyze ROI for medicine strip or bottle characteristics
        """
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for medicine packaging
        # Medicine strips often have metallic/silver backing
        silver_lower = np.array([0, 0, 180])
        silver_upper = np.array([255, 30, 255])
        
        # Medicine bottles often have white/transparent appearance
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([255, 30, 255])
        
        # Orange/amber bottles
        orange_lower = np.array([10, 100, 100])
        orange_upper = np.array([25, 255, 255])
        
        # Create masks
        silver_mask = cv2.inRange(hsv, silver_lower, silver_upper)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        
        # Calculate areas
        silver_area = cv2.countNonZero(silver_mask)
        white_area = cv2.countNonZero(white_mask)
        orange_area = cv2.countNonZero(orange_mask)
        
        total_area = roi.shape[0] * roi.shape[1]
        
        # Thresholds for detection
        strip_threshold = 0.15  # 15% of ROI should be silver for strip
        bottle_threshold = 0.20  # 20% of ROI should be white/orange for bottle
        
        result = {"medicine_object_detected": False, "object_type": None, "confidence": 0.0}
        
        if silver_area / total_area > strip_threshold:
            result = {
                "medicine_object_detected": True,
                "object_type": "medicine_strip",
                "confidence": min(0.95, (silver_area / total_area) / strip_threshold)
            }
        elif (white_area + orange_area) / total_area > bottle_threshold:
            result = {
                "medicine_object_detected": True,
                "object_type": "medicine_bottle",
                "confidence": min(0.95, ((white_area + orange_area) / total_area) / bottle_threshold)
            }
        
        return result
    
    def detect_medicine_intake_gesture(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Enhanced medicine intake detection with object recognition
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process all components
        hand_results = self.hands.process(rgb_image)
        pose_results = self.pose.process(rgb_image)
        face_results = self.face_detection.process(rgb_image)
        
        detection_result = {
            "hand_near_mouth": False,
            "head_tilted_back": False,
            "medicine_object_detected": False,
            "object_type": None,
            "confidence_scores": {},
            "timestamp": time.time()
        }
        
        # Check if hands and face are detected
        if not (hand_results.multi_hand_landmarks and face_results.detections):
            return detection_result
        
        # Get face center for reference
        face_detection = face_results.detections[0]
        face_bbox = face_detection.location_data.relative_bounding_box
        face_center_x = face_bbox.xmin + face_bbox.width / 2
        face_center_y = face_bbox.ymin + face_bbox.height / 2
        
        # Analyze each hand
        for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            # Get hand tip (index finger tip - landmark 8)
            hand_tip = hand_landmarks.landmark[8]
            
            # Calculate distance between hand and mouth area
            mouth_area_y = face_center_y + face_bbox.height * 0.3  # Lower part of face
            distance_to_mouth = np.sqrt(
                (hand_tip.x - face_center_x) ** 2 + 
                (hand_tip.y - mouth_area_y) ** 2
            )
            
            if distance_to_mouth < self.hand_mouth_distance_threshold:
                detection_result["hand_near_mouth"] = True
                detection_result["confidence_scores"]["hand_mouth_distance"] = 1 - (distance_to_mouth / self.hand_mouth_distance_threshold)
            
            # Check for medicine object in the other hand
            if len(hand_results.multi_hand_landmarks) == 2:
                other_hand_idx = 1 - hand_idx
                other_hand = hand_results.multi_hand_landmarks[other_hand_idx]
                
                # Detect medicine object in the other hand
                medicine_detection = self.detect_medicine_strip_bottle(image, other_hand)
                if medicine_detection["medicine_object_detected"]:
                    detection_result["medicine_object_detected"] = True
                    detection_result["object_type"] = medicine_detection["object_type"]
                    detection_result["confidence_scores"]["medicine_object"] = medicine_detection["confidence"]
        
        # Check head tilt using pose landmarks
        if pose_results.pose_landmarks:
            # Get nose and ear landmarks for head tilt analysis
            nose = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_ear = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
            
            # Calculate head tilt angle
            ear_center_y = (left_ear.y + right_ear.y) / 2
            head_tilt_ratio = (nose.y - ear_center_y)
            
            if head_tilt_ratio < -0.05:  # Head tilted back
                detection_result["head_tilted_back"] = True
                detection_result["confidence_scores"]["head_tilt"] = min(1.0, abs(head_tilt_ratio) * 10)
        
        return detection_result
    
    def process_video_file(self, video_path: str) -> Dict[str, Any]:
        """
        Process video file for medicine intake detection
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "error", "error": "Could not open video file"}
        
        detection_events = []
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp_sec = frame_count / fps
            
            # Perform detection every 5 frames for efficiency
            if frame_count % 5 == 0:
                detection = self.detect_medicine_intake_gesture(frame)
                
                # Check if this looks like medicine intake
                is_medicine_intake = (
                    detection["hand_near_mouth"] and 
                    detection["head_tilted_back"] and
                    detection["medicine_object_detected"]
                )
                
                if is_medicine_intake:
                    event = {
                        "timestamp_sec": timestamp_sec,
                        "frame_number": frame_count,
                        "detection_details": detection
                    }
                    detection_events.append(event)
        
        cap.release()
        
        # Determine final status
        if detection_events:
            return {
                "status": "taken",
                "events": detection_events,
                "total_frames": frame_count,
                "duration_sec": frame_count / fps
            }
        else:
            return {
                "status": "missed",
                "reason": "No complete medicine intake gesture detected",
                "total_frames": frame_count,
                "duration_sec": frame_count / fps
            }
    
    def start_live_monitoring(self, monitoring_duration: int = 60):
        """
        Start live webcam monitoring for medicine intake
        """
        if self.is_monitoring:
            logger.warning("Monitoring is already active")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._live_monitoring_worker,
            args=(monitoring_duration,)
        )
        self.monitoring_thread.start()
        logger.info(f"Started live monitoring for {monitoring_duration} seconds")
    
    def _live_monitoring_worker(self, duration: int):
        """
        Worker thread for live monitoring
        """
        cap = cv2.VideoCapture(0)  # Default webcam
        if not cap.isOpened():
            logger.error("Could not open webcam")
            self.is_monitoring = False
            return
        
        start_time = time.time()
        detection_events = []
        
        try:
            while self.is_monitoring and (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Perform detection
                detection = self.detect_medicine_intake_gesture(frame)
                
                # Check for medicine intake
                is_medicine_intake = (
                    detection["hand_near_mouth"] and 
                    detection["head_tilted_back"] and
                    detection["medicine_object_detected"]
                )
                
                if is_medicine_intake:
                    event = {
                        "timestamp": datetime.now().isoformat(),
                        "detection_details": detection
                    }
                    detection_events.append(event)
                    logger.info("Medicine intake detected during live monitoring!")
                
                # Optional: Display frame (comment out for headless operation)
                # cv2.imshow('Medicine Monitoring', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.is_monitoring = False
            
            # Save monitoring results
            self._save_monitoring_result(detection_events, duration)
    
    def _save_monitoring_result(self, events: List[Dict], duration: int):
        """
        Save monitoring results to file
        """
        result = {
            "monitoring_date": datetime.now().isoformat(),
            "duration_seconds": duration,
            "medicine_intake_detected": len(events) > 0,
            "detection_events": events,
            "total_detections": len(events)
        }
        
        # Save to JSON file
        filename = f"monitoring_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Monitoring result saved to {filename}")
    
    def schedule_daily_monitoring(self, time_str: str, duration: int = 60):
        """
        Schedule daily monitoring at specific time
        Format: "HH:MM" (24-hour format)
        """
        def monitoring_job():
            logger.info(f"Starting scheduled monitoring at {datetime.now()}")
            self.start_live_monitoring(duration)
        
        schedule.every().day.at(time_str).do(monitoring_job)
        self.scheduled_times.append((time_str, duration))
        logger.info(f"Scheduled daily monitoring at {time_str} for {duration} seconds")
    
    def run_scheduler(self):
        """
        Run the scheduler for daily monitoring
        Should be run in a separate thread
        """
        logger.info("Starting monitoring scheduler...")
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def stop_monitoring(self):
        """
        Stop live monitoring
        """
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join()
        logger.info("Live monitoring stopped")
    
    def release(self):
        """
        Clean up resources
        """
        self.stop_monitoring()
        if hasattr(self, 'hands'):
            self.hands.close()
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'face_detection'):
            self.face_detection.close()


# Main detection function for compatibility
def detect_medicine_intake(video_path: str) -> Dict[str, Any]:
    """
    Main function for medicine intake detection
    """
    detector = EnhancedMedicineDetector()
    try:
        result = detector.process_video_file(video_path)
        return result
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        detector.release()


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    detector = EnhancedMedicineDetector()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--live":
            # Start live monitoring
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            print(f"Starting live monitoring for {duration} seconds...")
            detector.start_live_monitoring(duration)
            
            # Wait for monitoring to complete
            while detector.is_monitoring:
                time.sleep(1)
                
        elif sys.argv[1] == "--schedule":
            # Schedule daily monitoring
            time_str = sys.argv[2] if len(sys.argv) > 2 else "09:00"
            duration = int(sys.argv[3]) if len(sys.argv) > 3 else 60
            
            print(f"Scheduling daily monitoring at {time_str} for {duration} seconds")
            detector.schedule_daily_monitoring(time_str, duration)
            
            # Run scheduler
            detector.run_scheduler()
            
        else:
            # Process video file
            result = detector.process_video_file(sys.argv[1])
            print(json.dumps(result, indent=2))
    else:
        print("Usage:")
        print("  python enhanced_detect_mediapipe_gesture.py video.mp4")
        print("  python enhanced_detect_mediapipe_gesture.py --live [duration_seconds]")
        print("  python enhanced_detect_mediapipe_gesture.py --schedule [HH:MM] [duration_seconds]")
    
    detector.release()