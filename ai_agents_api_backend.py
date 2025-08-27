"""
AI Agents API Backend for Medicine Intake Detection
FastAPI backend that uses AI agents for all functionality
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import tempfile
import time
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Import our AI agents system
from ai_agents_system import AIAgentsSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Agents Medicine Intake Detection API", version="3.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global AI agents system
ai_system = AIAgentsSystem()

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

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize AI agents system on startup"""
    try:
        await ai_system.start()
        logger.info("AI Agents System initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI Agents System: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup AI agents system on shutdown"""
    try:
        await ai_system.stop()
        logger.info("AI Agents System shutdown successfully")
    except Exception as e:
        logger.error(f"Error during AI Agents System shutdown: {e}")

@app.get("/")
async def root():
    return {
        "message": "AI Agents Medicine Intake Detection API",
        "version": "3.0.0",
        "features": [
            "AI Agent-based architecture",
            "Video file processing with AI analysis",
            "Live webcam monitoring",
            "Daily scheduling with AI coordination",
            "Enhanced medicine object detection",
            "Intelligent explanations and recommendations"
        ],
        "agents": [
            "Detection Agent - MediaPipe gesture detection",
            "Analysis Agent - AI-powered result analysis",
            "Scheduling Agent - Smart scheduling management",
            "Data Agent - Intelligent data storage and analytics",
            "Coordinator Agent - Workflow orchestration"
        ],
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with AI agents status"""
    try:
        system_status = await ai_system.get_system_status()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "ai_system_running": ai_system.running,
            "agents_status": system_status
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "ai_system_running": ai_system.running
        }

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    """
    Enhanced video upload with AI agents processing
    """
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Save uploaded file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    content = await file.read()
    temp_file.write(content)
    temp_file.close()
    
    try:
        # Process video with AI agents system
        start_time = time.time()
        result = await ai_system.process_video(temp_file.name)
        processing_time = time.time() - start_time
        
        # Add metadata
        if "detection" in result:
            result["detection"]["processing_time"] = round(processing_time, 2)
            result["detection"]["original_filename"] = file.filename
            result["detection"]["file_size_mb"] = round(len(content) / (1024 * 1024), 2)
            result["detection"]["processed_at"] = datetime.now().isoformat()
        
        # Ensure the result has the expected structure
        if result.get("workflow_status") == "completed":
            return result
        else:
            # Handle error cases
            return {
                "detection": result.get("detection", {"status": "error", "error": "Unknown error"}),
                "analysis": result.get("analysis", {"status": "error", "message": "Analysis failed"}),
                "explanation": result.get("explanation", "Unable to process video"),
                "reminder": result.get("reminder", "Please try again"),
                "workflow_status": "failed"
            }
        
    except Exception as e:
        logger.error(f"AI agents video processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI processing failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@app.post("/start-live-monitoring/")
async def start_live_monitoring(request: MonitoringRequest, background_tasks: BackgroundTasks):
    """
    Start live webcam monitoring using AI agents
    """
    try:
        result = await ai_system.start_live_monitoring(request.duration)
        
        if result.get("status") == "started":
            return {
                "status": "started",
                "duration": request.duration,
                "message": f"AI agents live monitoring started for {request.duration} seconds",
                "started_at": result.get("started_at"),
                "agents_coordinated": True
            }
        else:
            return {
                "status": "failed",
                "error": result.get("error", "Unknown error"),
                "message": "Failed to start AI agents monitoring"
            }
            
    except Exception as e:
        logger.error(f"AI agents live monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI monitoring failed: {str(e)}")

@app.post("/stop-monitoring/")
async def stop_monitoring():
    """
    Stop active monitoring using AI agents
    """
    try:
        # Send stop command through the detection agent
        task = {"type": "stop_monitoring"}
        result = await ai_system.coordinator.agents["detection"].execute_task(task)
        
        if result.get("status") == "stopped":
            return {
                "status": "stopped",
                "message": "AI agents monitoring stopped",
                "stopped_at": result.get("stopped_at")
            }
        else:
            return {
                "status": "failed",
                "error": result.get("error", "Unknown error")
            }
            
    except Exception as e:
        logger.error(f"Stop monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stop monitoring failed: {str(e)}")

@app.get("/monitoring-status/")
async def get_monitoring_status():
    """
    Get current monitoring status from AI agents
    """
    try:
        system_status = await ai_system.get_system_status()
        
        # Get scheduling information
        schedule_task = {"type": "get_schedules"}
        schedule_result = await ai_system.coordinator.agents["scheduling"].execute_task(schedule_task)
        
        return {
            "ai_system_running": ai_system.running,
            "agents_status": system_status,
            "scheduled_times": schedule_result.get("scheduled_times", []),
            "scheduler_active": schedule_result.get("scheduler_active", False),
            "current_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "ai_system_running": ai_system.running,
            "error": str(e),
            "current_time": datetime.now().isoformat()
        }

@app.post("/schedule-monitoring/")
async def schedule_monitoring(request: ScheduleRequest):
    """
    Schedule daily monitoring using AI agents
    """
    try:
        # Validate time format
        datetime.strptime(request.time, "%H:%M")
        
        # Use scheduling agent
        task = {
            "type": "add_schedule",
            "time": request.time,
            "duration": request.duration
        }
        result = await ai_system.coordinator.agents["scheduling"].execute_task(task)
        
        if result.get("status") == "scheduled":
            return {
                "status": "scheduled",
                "time": request.time,
                "duration": request.duration,
                "message": f"AI agents scheduled daily monitoring at {request.time} for {request.duration} seconds",
                "scheduled_at": result.get("scheduled_at"),
                "agent_coordinated": True
            }
        else:
            return {
                "status": "failed",
                "error": result.get("error", "Unknown error")
            }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid time format. Use HH:MM (24-hour format)")
    except Exception as e:
        logger.error(f"AI agents scheduling failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI scheduling failed: {str(e)}")

@app.get("/scheduled-times/")
async def get_scheduled_times():
    """
    Get all scheduled monitoring times from AI agents
    """
    try:
        task = {"type": "get_schedules"}
        result = await ai_system.coordinator.agents["scheduling"].execute_task(task)
        
        return ScheduleResponse(
            scheduled_times=result.get("scheduled_times", []),
            scheduler_active=result.get("scheduler_active", False)
        )
        
    except Exception as e:
        logger.error(f"Get schedules failed: {e}")
        return ScheduleResponse(
            scheduled_times=[],
            scheduler_active=False
        )

@app.delete("/clear-schedule/")
async def clear_schedule():
    """
    Clear all scheduled monitoring times using AI agents
    """
    try:
        task = {"type": "clear_schedules"}
        result = await ai_system.coordinator.agents["scheduling"].execute_task(task)
        
        return {
            "status": "cleared",
            "message": "All scheduled monitoring times cleared by AI agents",
            "cleared_at": result.get("cleared_at"),
            "agent_coordinated": True
        }
        
    except Exception as e:
        logger.error(f"Clear schedule failed: {e}")
        raise HTTPException(status_code=500, detail=f"Clear schedule failed: {str(e)}")

@app.delete("/remove-schedule/{time_str}")
async def remove_schedule(time_str: str):
    """
    Remove specific scheduled time using AI agents
    """
    try:
        task = {
            "type": "remove_schedule",
            "time": time_str
        }
        result = await ai_system.coordinator.agents["scheduling"].execute_task(task)
        
        if result.get("status") == "removed":
            return {
                "status": "removed",
                "time": time_str,
                "message": f"Schedule {time_str} removed by AI agents",
                "removed_at": result.get("removed_at")
            }
        else:
            return {
                "status": "failed",
                "error": result.get("error", "Unknown error")
            }
            
    except Exception as e:
        logger.error(f"Remove schedule failed: {e}")
        raise HTTPException(status_code=500, detail=f"Remove schedule failed: {str(e)}")

@app.get("/monitoring-results/")
async def get_monitoring_results():
    """
    Get recent monitoring results from AI agents
    """
    try:
        task = {"type": "get_results", "limit": 10}
        result = await ai_system.coordinator.agents["data"].execute_task(task)
        
        return {
            "results": result.get("results", []),
            "total_files": result.get("total_files", 0),
            "retrieved_at": result.get("retrieved_at"),
            "processed_by_ai_agents": True
        }
        
    except Exception as e:
        logger.error(f"Get results failed: {e}")
        return {
            "results": [],
            "total_files": 0,
            "error": str(e),
            "retrieved_at": datetime.now().isoformat()
        }

@app.get("/analytics/")
async def get_analytics():
    """
    Get analytics from AI agents
    """
    try:
        result = await ai_system.get_analytics()
        
        return {
            **result,
            "generated_by_ai_agents": True,
            "analytics_enhanced": True
        }
        
    except Exception as e:
        logger.error(f"Analytics failed: {e}")
        return {
            "total_sessions": 0,
            "successful_detections": 0,
            "success_rate": 0,
            "recent_sessions": 0,
            "error": str(e),
            "analytics_generated_at": datetime.now().isoformat()
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

@app.post("/export-results/")
async def export_results():
    """
    Export all results using AI agents
    """
    try:
        task = {"type": "export_data"}
        result = await ai_system.coordinator.agents["data"].execute_task(task)
        
        if result.get("status") == "exported":
            return {
                "status": "exported",
                "filename": result.get("filename"),
                "message": "Data exported successfully by AI agents",
                "exported_at": result.get("exported_at"),
                "download_url": f"/download-result/{result.get('filename')}"
            }
        else:
            return {
                "status": "failed",
                "error": result.get("error", "Export failed")
            }
            
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.post("/test-detection/")
async def test_detection():
    """
    Test AI agents system with mock data
    """
    try:
        # Create mock detection result
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
            "processed_at": datetime.now().isoformat(),
            "processed_by": "detection_agent"
        }
        
        # Process through analysis agent
        analysis_task = {"type": "analyze_detection", "detection_result": mock_result}
        analysis_result = await ai_system.coordinator.agents["analysis"].execute_task(analysis_task)
        
        # Generate explanation
        explanation_task = {"type": "generate_explanation", "detection_result": mock_result}
        explanation_result = await ai_system.coordinator.agents["analysis"].execute_task(explanation_task)
        
        # Generate reminder
        reminder = ai_system.coordinator._generate_reminder(analysis_result)
        
        return {
            "detection": mock_result,
            "analysis": analysis_result,
            "explanation": explanation_result.get("explanation"),
            "reminder": reminder,
            "workflow_status": "completed",
            "test_mode": True,
            "processed_by_ai_agents": True
        }
        
    except Exception as e:
        logger.error(f"Test detection failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "test_mode": True,
            "workflow_status": "failed"
        }

@app.get("/agents-status/")
async def get_agents_status():
    """
    Get detailed status of all AI agents
    """
    try:
        status = await ai_system.get_system_status()
        
        # Add additional information
        enhanced_status = {
            "system_running": ai_system.running,
            "agents": status,
            "capabilities": {
                "detection": ["video_analysis", "gesture_detection", "object_detection"],
                "analysis": ["result_analysis", "explanation_generation", "confidence_scoring"],
                "scheduling": ["daily_scheduling", "monitoring_timing", "schedule_management"],
                "data": ["data_storage", "result_retrieval", "analytics"],
                "coordinator": ["workflow_orchestration", "agent_coordination", "task_distribution"]
            },
            "communication_protocol": "Agent Message System",
            "ai_enhanced": True,
            "status_retrieved_at": datetime.now().isoformat()
        }
        
        return enhanced_status
        
    except Exception as e:
        logger.error(f"Agents status failed: {e}")
        return {
            "system_running": ai_system.running,
            "error": str(e),
            "status_retrieved_at": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting AI Agents Medicine Intake Detection API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
