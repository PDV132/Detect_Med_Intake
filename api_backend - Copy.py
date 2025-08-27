from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import logging
from typing import Tuple, Any  # Added for Python 3.8 compatibility
from agents import multi_agent_workflow
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medicine Intake Detection API", version="1.0.0")

# Add CORS middleware for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def validate_video_file(filename: str) -> Tuple[bool, str]:  # Fixed for Python 3.8
    """Validate video file format"""
    allowed_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    if not filename.lower().endswith(allowed_extensions):
        return False, f"Invalid file format. Allowed: {', '.join(allowed_extensions)}"
    return True, "Valid format"

def validate_video_content(video_path: str) -> Tuple[bool, str]:  # Fixed for Python 3.8
    """Validate video content using OpenCV"""
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available - skipping video content validation")
        return True, "Video validation skipped (OpenCV not available)"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Cannot open video file - file may be corrupted"
    
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0:
        cap.release()
        return False, "Invalid video - no frame rate detected"
    
    duration = frame_count / fps
    cap.release()
    
    if duration < 1:
        return False, "Video too short (minimum 1 second required)"
    if duration > 120:  # 2 minutes max
        return False, "Video too long (maximum 2 minutes allowed)"
    
    return True, f"Valid video - Duration: {duration:.1f}s"

@app.get("/")
async def root():
    return {"message": "Medicine Intake Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "medicine-detection-api"}

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    logger.info(f"Received video upload: {file.filename}")
    
    # Validate file format
    is_valid_format, format_message = validate_video_file(file.filename)
    if not is_valid_format:
        logger.error(f"Invalid file format: {file.filename}")
        raise HTTPException(status_code=400, detail=format_message)
    
    video_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # Save uploaded file
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Video saved to: {video_path}")
        
        # Validate video content
        is_valid_content, content_message = validate_video_content(video_path)
        if not is_valid_content:
            os.remove(video_path)
            logger.error(f"Invalid video content: {content_message}")
            raise HTTPException(status_code=400, detail=content_message)
        
        logger.info(f"Video validation passed: {content_message}")
        
        # Process video through multi-agent workflow
        logger.info("Starting multi-agent workflow...")
        result = multi_agent_workflow(video_path)
        logger.info("Multi-agent workflow completed successfully")
        
        # Clean up uploaded file
        if os.path.exists(video_path):
            os.remove(video_path)
            logger.info(f"Cleaned up temporary file: {video_path}")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(video_path):
            os.remove(video_path)
            logger.info(f"Cleaned up file after error: {video_path}")
        
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Video processing failed: {str(e)}"
        )

# Additional endpoint for testing
@app.post("/test-detection/")
async def test_detection():
    """Test endpoint that returns a mock detection result"""
    mock_result = {
        "detection": {
            "status": "taken",
            "events": [
                {
                    "timestamp_sec": 5.2,
                    "head_sideways": False,
                    "head_bent_backward": True
                }
            ]
        },
        "analysis": {
            "status": "taken",
            "message": "Medicine intake gesture detected successfully."
        },
        "explanation": "Medicine intake gesture detected successfully.",
        "reminder": "✅ Good job! Medicine intake confirmed."
    }
    return JSONResponse(content=mock_result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)