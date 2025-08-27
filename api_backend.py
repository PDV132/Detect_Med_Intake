"""
FastAPI endpoint for video processing with face and hand masking
Add this to your api_backend.py file
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import os
import tempfile
import time
from typing import Optional

app = FastAPI(title="Medicine Intake Detection API")

# Add CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create uploads directory
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Medicine Intake Detection API", "status": "running"}

@app.post("/process-video")
async def process_video(
    file: UploadFile = File(...),
    mask_face: bool = Form(True),
    mask_hands: bool = Form(True),
    detect_medicine: bool = Form(False)
):
    """
    Process video with face and hand masking
    """
    start_time = time.time()
    
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Save uploaded file
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    content = await file.read()
    temp_input.write(content)
    temp_input.close()
    
    try:
        # Process video
        processed_path = process_video_with_masking(
            temp_input.name,
            mask_face=mask_face,
            mask_hands=mask_hands,
            detect_medicine=detect_medicine
        )
        
        processing_time = time.time() - start_time
        
        # Return results
        return {
            "status": "success",
            "processed_video_url": f"/download/{os.path.basename(processed_path)}",
            "face_masking": mask_face,
            "hand_masking": mask_hands,
            "medicine_detection": detect_medicine,
            "processing_time": round(processing_time, 2),
            "original_filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_input.name):
            os.unlink(temp_input.name)

@app.get("/download/{filename}")
async def download_processed_video(filename: str):
    """
    Download processed video file
    """
    file_path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type='video/mp4',
        filename=filename
    )

def process_video_with_masking(input_path: str, mask_face: bool = True, mask_hands: bool = True, detect_medicine: bool = False):
    """
    Process video with MediaPipe face and hand detection/masking
    """
    # Open video
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    output_path = os.path.join(PROCESSED_DIR, f"processed_{int(time.time())}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize MediaPipe models
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    frame_count = 0
    face_detections = 0
    hand_detections = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process face detection
            if mask_face:
                face_results = face_detection.process(rgb_frame)
                if face_results.detections:
                    face_detections += len(face_results.detections)
                    for detection in face_results.detections:
                        # Get bounding box
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        
                        # Convert to pixel coordinates
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width_box = int(bbox.width * w)
                        height_box = int(bbox.height * h)
                        
                        # Apply blur to face region
                        face_region = frame[y:y+height_box, x:x+width_box]
                        if face_region.size > 0:
                            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                            frame[y:y+height_box, x:x+width_box] = blurred_face
            
            # Process hand detection
            if mask_hands:
                hand_results = hands.process(rgb_frame)
                if hand_results.multi_hand_landmarks:
                    hand_detections += len(hand_results.multi_hand_landmarks)
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        # Get hand bounding box
                        h, w, _ = frame.shape
                        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                        
                        x_min, x_max = int(min(x_coords)), int(max(x_coords))
                        y_min, y_max = int(min(y_coords)), int(max(y_coords))
                        
                        # Add padding
                        padding = 20
                        x_min = max(0, x_min - padding)
                        y_min = max(0, y_min - padding)
                        x_max = min(w, x_max + padding)
                        y_max = min(h, y_max + padding)
                        
                        # Apply blur to hand region
                        hand_region = frame[y_min:y_max, x_min:x_max]
                        if hand_region.size > 0:
                            blurred_hand = cv2.GaussianBlur(hand_region, (51, 51), 15)
                            frame[y_min:y_max, x_min:x_max] = blurred_hand
            
            # Write processed frame
            out.write(frame)
    
    finally:
        # Release everything
        cap.release()
        out.release()
        face_detection.close()
        hands.close()
    
    print(f"Processed {frame_count} frames")
    print(f"Face detections: {face_detections}")
    print(f"Hand detections: {hand_detections}")
    
    return output_path

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)