# enhanced_api_backend.py
"""
Enhanced FastAPI backend with live monitoring, scheduling, and medicine object detection
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
import os
import tempfile
import time
import json
import threading
import schedule
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Import our enhanced detection module
try:
    from enhanced_detect_mediapipe_gesture import EnhancedMedicineDetector
except ImportError:
    # Fallback if module not available
    class EnhancedMedicineDetector:
        def __init__(self):
            pass
        def process_video_file(self, path):
            return {"status": "error", "error": "Detection module not available"}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Medicine Intake Detection API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector = EnhancedMedicineDetector()
scheduler_thread = None
monitoring_active = False

# Create directories
UPLOAD_DIR = "uploads"
RESULTS_DIR = "monitoring_results"
for dir_path in [UPLOAD_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Pydantic models
class ScheduleRequest(BaseModel):
    time: str  # Format: "HH:MM"
    duration: int = 60  # Duration in seconds
    enabled: bool = True

class MonitoringRequest(BaseModel):
    duration: int = 60  # Duration in seconds

class ScheduleResponse(BaseModel):
    scheduled_times: List[Dict[str, Any]]
    scheduler_active: bool

@app.get("/")
async def root():
    return {
        "message": "Enhanced Medicine Intake Detection API",
        "version": "2.0.0",
        "features": [
            "Video file processing",
            "Live webcam monitoring",
            "Daily scheduling",
            "Medicine object detection",
            "Hand gesture recognition"
        ],
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "monitoring_active": monitoring_active,
        "scheduler_active": scheduler_thread is not None and scheduler_thread.is_alive()
    }

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    """
    Enhanced video upload with medicine object detection
    """
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Save uploaded file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    content = await file.read()
    temp_file.write(content)
    temp_file.close()
    
    try:
        # Process video with enhanced detection
        start_time = time.time()
        result = detector.process_video_file(temp_file.name)
        processing_time = time.time() - start_time
        
        # Add metadata
        result["processing_time"] = round(processing_time, 2)
        result["original_filename"] = file.filename
        result["file_size_mb"] = round(len(content) / (1024 * 1024), 2)
        result["processed_at"] = datetime.now().isoformat()
        
        # Enhanced analysis for response
        analysis = analyze_detection_result(result)
        
        return {
            "detection": result,
            "analysis": analysis,
            "explanation": generate_explanation(result, analysis),
            "reminder": generate_reminder(result),
            "workflow_status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@app.post("/start-live-monitoring/")
async def start_live_monitoring(request: MonitoringRequest, background_tasks: BackgroundTasks):
    """
    Start live webcam monitoring
    """
    global monitoring_active
    
    if monitoring_active:
        return {"status": "already_active", "message": "Monitoring is already running"}
    
    # Start monitoring in background
    background_tasks.add_task(run_live_monitoring, request.duration)
    
    return {
        "status": "started",
        "duration": request.duration,
        "message": f"Live monitoring started for {request.duration} seconds",
        "started_at": datetime.now().isoformat()
    }

@app.post("/stop-monitoring/")
async def stop_monitoring():
    """
    Stop active monitoring
    """
    global monitoring_active
    
    if not monitoring_active:
        return {"status": "not_active", "message": "No active monitoring to stop"}
    
    detector.stop_monitoring()
    monitoring_active = False
    
    return {
        "status": "stopped",
        "message": "Live monitoring stopped",
        "stopped_at": datetime.now().isoformat()
    }

@app.get("/monitoring-status/")
async def get_monitoring_status():
    """
    Get current monitoring status
    """
    return {
        "monitoring_active": monitoring_active,
        "scheduled_times": detector.scheduled_times,
        "scheduler_active": scheduler_thread is not None and scheduler_thread.is_alive(),
        "current_time": datetime.now().isoformat()
    }

@app.post("/schedule-monitoring/")
async def schedule_monitoring(request: ScheduleRequest):
    """
    Schedule daily monitoring at specific time
    """
    global scheduler_thread
    
    try:
        # Validate time format
        datetime.strptime(request.time, "%H:%M")
        
        # Add to schedule
        detector.schedule_daily_monitoring(request.time, request.duration)
        
        # Start scheduler thread if not running
        if scheduler_thread is None or not scheduler_thread.is_alive():
            scheduler_thread = threading.Thread(target=detector.run_scheduler, daemon=True)
            scheduler_thread.start()
            logger.info("Scheduler thread started")
        
        return {
            "status": "scheduled",
            "time": request.time,
            "duration": request.duration,
            "message": f"Daily monitoring scheduled at {request.time} for {request.duration} seconds",
            "scheduled_at": datetime.now().isoformat()
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid time format. Use HH:MM (24-hour format)")
    except Exception as e:
        logger.error(f"Scheduling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scheduling failed: {str(e)}")

@app.get("/scheduled-times/")
async def get_scheduled_times():
    """
    Get all scheduled monitoring times
    """
    return ScheduleResponse(
        scheduled_times=[
            {"time": time_str, "duration": duration} 
            for time_str, duration in detector.scheduled_times
        ],
        scheduler_active=scheduler_thread is not None and scheduler_thread.is_alive()
    )

@app.delete("/clear-schedule/")
async def clear_schedule():
    """
    Clear all scheduled monitoring times
    """
    schedule.clear()
    detector.scheduled_times.clear()
    
    return {
        "status": "cleared",
        "message": "All scheduled monitoring times cleared",
        "cleared_at": datetime.now().isoformat()
    }

@app.get("/monitoring-results/")
async def get_monitoring_results():
    """
    Get recent monitoring results
    """
    results_dir = Path(RESULTS_DIR)
    result_files = list(results_dir.glob("monitoring_result_*.json"))
    
    # Sort by creation time, most recent first
    result_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
    
    results = []
    for file_path in result_files[:10]:  # Return last 10 results
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
                result["filename"] = file_path.name
                results.append(result)
        except Exception as e:
            logger.warning(f"Could not read result file {file_path}: {e}")
    
    return {
        "results": results,
        "total_files": len(result_files),
        "retrieved_at": datetime.now().isoformat()
    }

@app.get("/download-result/{filename}")
async def download_result(filename: str):
    """
    Download specific monitoring result file
    """
    file_path = Path(RESULTS_DIR) / filename
    
    if not file_path.exists() or not filename.endswith('.json'):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        file_path,
        media_type='application/json',
        filename=filename
    )

@app.post("/test-detection/")
async def test_detection():
    """
    Test detection with mock data
    """
    mock_result = {
        "status": "taken",
        "events": [
            {
                "timestamp_sec": 3.5,
                "frame_number": 105,
                "detection_details": {
                    "hand_near_mouth": True,
                    "head_tilted_back": True,
                    "medicine_object_detected": True,
                    "object_type": "medicine_strip",
                    "confidence_scores": {
                        "hand_mouth_distance": 0.85,
                        "head_tilt": 0.72,
                        "medicine_object": 0.78
                    }
                }
            }
        ],
        "total_frames": 150,
        "duration_sec": 5.0,
        "processing_time": 1.2,
        "processed_at": datetime.now().isoformat()
    }
    
    analysis = analyze_detection_result(mock_result)
    
    return {
        "detection": mock_result,
        "analysis": analysis,
        "explanation": generate_explanation(mock_result, analysis),
        "reminder": generate_reminder(mock_result),
        "workflow_status": "completed",
        "test_mode": True
    }

# Helper functions
async def run_live_monitoring(duration: int):
    """
    Background task for live monitoring
    """
    global monitoring_active
    monitoring_active = True
    
    try:
        detector.start_live_monitoring(duration)
        
        # Wait for monitoring to complete
        while detector.is_monitoring:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"Live monitoring failed: {e}")
    finally:
        monitoring_active = False

def analyze_detection_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze detection result and provide insights
    """
    status = result.get("status", "unknown")
    
    if status == "taken":
        events = result.get("events", [])
        if events:
            # Analyze confidence scores from the first event
            detection_details = events[0].get("detection_details", {})
            confidence_scores = detection_details.get("confidence_scores", {})
            
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0
            
            analysis = {
                "status": "taken",
                "confidence": "high" if avg_confidence > 0.7 else "medium",
                "medicine_object_detected": detection_details.get("medicine_object_detected", False),
                "object_type": detection_details.get("object_type"),
                "analysis_scores": confidence_scores
            }
        else:
            analysis = {"status": "taken", "confidence": "medium"}
    
    elif status == "error":
        analysis = {
            "status": "error",
            "confidence": "none",
            "message": result.get("error", "Unknown error occurred")
        }
    
    else:  # missed
        analysis = {
            "status": "missed",
            "confidence": "low",
            "message": "Medicine intake gesture not detected",
            "possible_reasons": [
                "Hand-to-mouth movement not detected",
                "Head tilt backward not observed", 
                "Medicine object (strip/bottle) not visible",
                "Poor lighting or camera angle"
            ]
        }
    
    return analysis

def generate_explanation(detection_result: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    """
    Generate human-readable explanation of detection results
    """
    status = detection_result.get("status", "unknown")
    
    if status == "taken":
        events = detection_result.get("events", [])
        if events:
            event = events[0]
            details = event.get("detection_details", {})
            
            explanation = f"✅ Medicine intake successfully detected at {event.get('timestamp_sec', 0):.1f} seconds. "
            
            if details.get("medicine_object_detected"):
                object_type = details.get("object_type", "medicine object")
                explanation += f"The system identified a {object_type.replace('_', ' ')} being held, "
            
            explanation += "along with proper hand-to-mouth movement and head positioning typical of taking medication."
            
            # Add confidence information
            confidence_scores = details.get("confidence_scores", {})
            if confidence_scores:
                highest_confidence = max(confidence_scores.values())
                explanation += f" Detection confidence: {highest_confidence:.0%}."
        else:
            explanation = "Medicine intake detected but with limited detail information."
    
    elif status == "error":
        explanation = f"❌ Detection failed due to an error: {detection_result.get('error', 'Unknown error')}. Please try again with a clear video file."
    
    else:  # missed
        explanation = "⚠️ Medicine intake not detected in the video. "
        
        # Provide specific guidance based on what was missing
        possible_reasons = []
        if not detection_result.get("hand_near_mouth", True):
            possible_reasons.append("ensure your hand clearly moves to your mouth")
        if not detection_result.get("head_tilted_back", True):
            possible_reasons.append("tilt your head slightly back when swallowing")
        if not detection_result.get("medicine_object_detected", True):
            possible_reasons.append("hold the medicine strip or bottle clearly visible in your other hand")
        
        if possible_reasons:
            explanation += "For better detection: " + ", ".join(possible_reasons) + ". "
        
        explanation += "Ensure good lighting and that your upper body is clearly visible to the camera."
    
    return explanation

def generate_reminder(detection_result: Dict[str, Any]) -> str:
    """
    Generate appropriate reminder message
    """
    status = detection_result.get("status", "unknown")
    
    if status == "taken":
        return "✅ Excellent! Medicine intake confirmed. Keep maintaining your medication schedule!"
    elif status == "error":
        return "❌ Unable to analyze the recording. Please try again with a clear video or check your camera."
    else:  # missed
        return "⚠️ Medicine intake not detected. Please ensure you take your medication as prescribed and record it properly for tracking."

# Add async import for background tasks
import asyncio

if __name__ == "__main__":
    import uvicorn
    
    # Start the scheduler thread
    scheduler_thread = threading.Thread(target=detector.run_scheduler, daemon=True)
    scheduler_thread.start()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)