"""
AI Agents System for Medicine Intake Detection
Converts the existing system to use AI agents for all functionality
"""

import os
import json
import logging
import asyncio
import threading
import time
import schedule
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import cv2
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI = bool(OPENAI_API_KEY)

if USE_OPENAI:
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
        logger.info("OpenAI LLM initialized successfully")
    except ImportError:
        logger.warning("langchain_openai not available, using fallback")
        USE_OPENAI = False
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI LLM: {e}")
        USE_OPENAI = False

# Agent Communication Protocol
class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    COORDINATION = "coordination"

@dataclass
class AgentMessage:
    sender: str
    receiver: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime
    message_id: str

class AgentStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"

# Base Agent Class
class BaseAgent:
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.status = AgentStatus.IDLE
        self.message_queue = asyncio.Queue()
        self.capabilities = []
        self.running = False
        self.task_history = []
        
    async def start(self):
        """Start the agent"""
        self.running = True
        self.status = AgentStatus.IDLE
        logger.info(f"Agent {self.name} started")
        
    async def stop(self):
        """Stop the agent"""
        self.running = False
        self.status = AgentStatus.OFFLINE
        logger.info(f"Agent {self.name} stopped")
        
    async def send_message(self, receiver: str, message_type: MessageType, content: Dict[str, Any]):
        """Send message to another agent"""
        message = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(),
            message_id=f"{self.agent_id}_{int(time.time())}"
        )
        # In a real system, this would go through a message broker
        logger.info(f"{self.name} sending message to {receiver}: {message_type.value}")
        return message
        
    async def process_message(self, message: AgentMessage):
        """Process incoming message"""
        logger.info(f"{self.name} received message from {message.sender}: {message.message_type.value}")
        
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task - to be implemented by subclasses"""
        raise NotImplementedError

# Detection Agent - Handles MediaPipe detection
class DetectionAgent(BaseAgent):
    def __init__(self):
        super().__init__("detection_agent", "Detection Agent")
        self.capabilities = ["video_analysis", "gesture_detection", "object_detection"]
        self.detector = None
        self._initialize_detector()
        
    def _initialize_detector(self):
        """Initialize the strict detector with three-condition validation"""
        try:
            from enhanced_medicine_detection import StrictMedicineDetector
            self.detector = StrictMedicineDetector()
            logger.info("Strict medicine detector initialized successfully")
        except ImportError:
            try:
                from enhanced_detect_mediapipe_gesture import EnhancedMedicineDetector
                self.detector = EnhancedMedicineDetector()
                logger.info("Enhanced detector initialized as fallback")
            except ImportError:
                logger.warning("No detector available, using fallback")
                self.detector = None
            
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute detection task"""
        self.status = AgentStatus.BUSY
        
        try:
            task_type = task.get("type")
            
            if task_type == "analyze_video":
                return await self._analyze_video(task.get("video_path"))
            elif task_type == "start_live_monitoring":
                return await self._start_live_monitoring(task.get("duration", 60))
            elif task_type == "stop_monitoring":
                return await self._stop_monitoring()
            else:
                return {"status": "error", "error": f"Unknown task type: {task_type}"}
                
        except Exception as e:
            logger.error(f"Detection task failed: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            self.status = AgentStatus.IDLE
            
    async def _analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Analyze video file with strict three-condition validation"""
        if not self.detector:
            return {"status": "error", "error": "Detector not available"}
            
        try:
            # Use strict validation if available
            if hasattr(self.detector, 'process_video_with_strict_validation'):
                result = self.detector.process_video_with_strict_validation(video_path)
                logger.info("Using strict three-condition validation")
            else:
                result = self.detector.process_video_file(video_path)
                logger.info("Using fallback detection method")
            
            # Add metadata
            result["processed_by"] = self.agent_id
            result["processed_at"] = datetime.now().isoformat()
            result["detection_method"] = "strict_validation" if hasattr(self.detector, 'process_video_with_strict_validation') else "enhanced_detection"
            
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}
            
    async def _start_live_monitoring(self, duration: int) -> Dict[str, Any]:
        """Start live monitoring"""
        if not self.detector:
            return {"status": "error", "error": "Detector not available"}
            
        try:
            self.detector.start_live_monitoring(duration)
            return {
                "status": "started",
                "duration": duration,
                "started_at": datetime.now().isoformat()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
            
    async def _stop_monitoring(self) -> Dict[str, Any]:
        """Stop live monitoring"""
        if not self.detector:
            return {"status": "error", "error": "Detector not available"}
            
        try:
            self.detector.stop_monitoring()
            return {
                "status": "stopped",
                "stopped_at": datetime.now().isoformat()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Analysis Agent - Handles AI analysis and explanations
class AnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("analysis_agent", "Analysis Agent")
        self.capabilities = ["result_analysis", "explanation_generation", "confidence_scoring"]
        
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis task"""
        self.status = AgentStatus.BUSY
        
        try:
            task_type = task.get("type")
            
            if task_type == "analyze_detection":
                return await self._analyze_detection(task.get("detection_result"))
            elif task_type == "generate_explanation":
                return await self._generate_explanation(task.get("detection_result"))
            elif task_type == "assess_confidence":
                return await self._assess_confidence(task.get("detection_result"))
            else:
                return {"status": "error", "error": f"Unknown task type: {task_type}"}
                
        except Exception as e:
            logger.error(f"Analysis task failed: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            self.status = AgentStatus.IDLE
            
    async def _analyze_detection(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze detection results"""
        status = detection_result.get("status", "unknown")
        
        if status == "taken":
            events = detection_result.get("events", [])
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
                    "analysis_scores": confidence_scores,
                    "analyzed_by": self.agent_id
                }
            else:
                analysis = {"status": "taken", "confidence": "medium", "analyzed_by": self.agent_id}
        
        elif status == "error":
            analysis = {
                "status": "error",
                "confidence": "none",
                "message": detection_result.get("error", "Unknown error occurred"),
                "analyzed_by": self.agent_id
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
                ],
                "analyzed_by": self.agent_id
            }
        
        return analysis
        
    async def _generate_explanation(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable explanation"""
        status = detection_result.get("status", "unknown")
        
        if USE_OPENAI:
            try:
                prompt = f"""
                Based on this medicine intake detection result, provide a clear, helpful explanation:
                
                Detection Result: {json.dumps(detection_result, indent=2)}
                
                Please provide:
                1. A clear explanation of what was detected or not detected
                2. Specific feedback about the detection quality
                3. Actionable suggestions for improvement if needed
                
                Keep it concise, encouraging, and helpful (2-3 sentences).
                """
                
                response = await self._query_llm(prompt)
                explanation = response.strip()
                
            except Exception as e:
                logger.error(f"LLM explanation failed: {e}")
                explanation = self._generate_fallback_explanation(detection_result)
        else:
            explanation = self._generate_fallback_explanation(detection_result)
            
        return {
            "explanation": explanation,
            "generated_by": self.agent_id,
            "generated_at": datetime.now().isoformat()
        }
        
    def _generate_fallback_explanation(self, detection_result: Dict[str, Any]) -> str:
        """Generate fallback explanation without LLM"""
        status = detection_result.get("status", "unknown")
        validation_method = detection_result.get("validation_method", "")
        
        if status == "taken":
            events = detection_result.get("events", [])
            if events and validation_method == "strict_three_condition_check":
                # Strict validation success
                event = events[0]
                validation_details = event.get("validation_details", {})
                
                explanation = f"✅ Medicine intake successfully validated at {event.get('timestamp_sec', 0):.1f} seconds using strict three-condition check:\n"
                
                # Detail each validated condition
                if validation_details.get("condition_1_medicine_in_hand"):
                    med_details = validation_details.get("medicine_object_details", {})
                    object_type = med_details.get("object_type", "medicine object")
                    explanation += f"• {object_type.replace('_', ' ').title()} detected in hand\n"
                
                if validation_details.get("condition_2_hand_near_mouth"):
                    hand_details = validation_details.get("hand_mouth_details", {})
                    finger = hand_details.get("closest_finger", "finger")
                    explanation += f"• {finger.title()} positioned near mouth\n"
                
                if validation_details.get("condition_3_head_tilted_back"):
                    head_details = validation_details.get("head_tilt_details", {})
                    angle = head_details.get("tilt_angle_estimate", 0)
                    explanation += f"• Head tilted back (~{angle:.0f}° angle)\n"
                
                explanation += f"\nTotal valid frames: {detection_result.get('total_valid_frames', 0)}"
                
            elif events:
                # Regular detection success
                event = events[0]
                explanation = f"✅ Medicine intake detected at {event.get('timestamp_sec', 0):.1f} seconds with standard validation."
            else:
                explanation = "✅ Medicine intake detected but with limited detail information."
        
        elif status == "error":
            explanation = f"❌ Detection failed due to an error: {detection_result.get('error', 'Unknown error')}. Please try again with a clear video file."
        
        else:  # missed
            if validation_method == "strict_three_condition_check":
                failure_details = detection_result.get("failure_details", [])
                explanation = "⚠️ Medicine intake not validated. The strict three-condition check failed:\n"
                
                for detail in failure_details:
                    explanation += f"• {detail}\n"
                
                explanation += "\n📋 Required conditions:\n"
                conditions_required = detection_result.get("conditions_required", [])
                for condition in conditions_required:
                    explanation += f"• {condition}\n"
                
                explanation += "\n💡 Tips for better detection:\n"
                explanation += "• Hold medicine strip/bottle clearly visible in one hand\n"
                explanation += "• Bring other hand close to mouth (fingers near lips)\n"
                explanation += "• Tilt head back slightly when swallowing\n"
                explanation += "• Ensure good lighting and clear camera view"
            else:
                explanation = "⚠️ Medicine intake not detected in the video. Ensure good lighting and that your upper body is clearly visible to the camera."
        
        return explanation
        
    async def _assess_confidence(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess confidence in detection results"""
        status = detection_result.get("status", "unknown")
        
        if status == "taken":
            events = detection_result.get("events", [])
            if events:
                detection_details = events[0].get("detection_details", {})
                confidence_scores = detection_details.get("confidence_scores", {})
                
                if confidence_scores:
                    avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
                    confidence_level = "high" if avg_confidence > 0.7 else "medium" if avg_confidence > 0.4 else "low"
                else:
                    confidence_level = "medium"
                    avg_confidence = 0.6
            else:
                confidence_level = "medium"
                avg_confidence = 0.5
        elif status == "error":
            confidence_level = "none"
            avg_confidence = 0.0
        else:
            confidence_level = "low"
            avg_confidence = 0.2
            
        return {
            "confidence_level": confidence_level,
            "confidence_score": avg_confidence,
            "assessed_by": self.agent_id
        }
        
    async def _query_llm(self, prompt: str) -> str:
        """Query the LLM"""
        if USE_OPENAI:
            try:
                response = llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                logger.error(f"LLM query failed: {e}")
                raise
        else:
            raise Exception("LLM not available")

# Scheduling Agent - Handles scheduling and timing
class SchedulingAgent(BaseAgent):
    def __init__(self):
        super().__init__("scheduling_agent", "Scheduling Agent")
        self.capabilities = ["daily_scheduling", "monitoring_timing", "schedule_management"]
        self.scheduled_times = []
        self.scheduler_thread = None
        self.scheduler_running = False
        
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scheduling task"""
        self.status = AgentStatus.BUSY
        
        try:
            task_type = task.get("type")
            
            if task_type == "add_schedule":
                return await self._add_schedule(task.get("time"), task.get("duration", 60))
            elif task_type == "remove_schedule":
                return await self._remove_schedule(task.get("time"))
            elif task_type == "clear_schedules":
                return await self._clear_schedules()
            elif task_type == "get_schedules":
                return await self._get_schedules()
            elif task_type == "start_scheduler":
                return await self._start_scheduler()
            elif task_type == "stop_scheduler":
                return await self._stop_scheduler()
            else:
                return {"status": "error", "error": f"Unknown task type: {task_type}"}
                
        except Exception as e:
            logger.error(f"Scheduling task failed: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            self.status = AgentStatus.IDLE
            
    async def _add_schedule(self, time_str: str, duration: int) -> Dict[str, Any]:
        """Add a scheduled monitoring time"""
        try:
            # Validate time format
            datetime.strptime(time_str, "%H:%M")
            
            # Add to schedule
            schedule.every().day.at(time_str).do(self._scheduled_monitoring_job, duration)
            self.scheduled_times.append((time_str, duration))
            
            # Start scheduler if not running
            if not self.scheduler_running:
                await self._start_scheduler()
            
            return {
                "status": "scheduled",
                "time": time_str,
                "duration": duration,
                "scheduled_at": datetime.now().isoformat()
            }
            
        except ValueError:
            return {"status": "error", "error": "Invalid time format. Use HH:MM"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
            
    async def _remove_schedule(self, time_str: str) -> Dict[str, Any]:
        """Remove a scheduled time"""
        try:
            # Remove from our list
            self.scheduled_times = [(t, d) for t, d in self.scheduled_times if t != time_str]
            
            # Clear and rebuild schedule
            schedule.clear()
            for time_str, duration in self.scheduled_times:
                schedule.every().day.at(time_str).do(self._scheduled_monitoring_job, duration)
            
            return {
                "status": "removed",
                "time": time_str,
                "removed_at": datetime.now().isoformat()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
            
    async def _clear_schedules(self) -> Dict[str, Any]:
        """Clear all schedules"""
        schedule.clear()
        self.scheduled_times.clear()
        
        return {
            "status": "cleared",
            "cleared_at": datetime.now().isoformat()
        }
        
    async def _get_schedules(self) -> Dict[str, Any]:
        """Get all scheduled times"""
        return {
            "scheduled_times": [
                {"time": time_str, "duration": duration} 
                for time_str, duration in self.scheduled_times
            ],
            "scheduler_active": self.scheduler_running
        }
        
    async def _start_scheduler(self) -> Dict[str, Any]:
        """Start the scheduler thread"""
        if not self.scheduler_running:
            self.scheduler_running = True
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            logger.info("Scheduler thread started")
            
        return {
            "status": "started",
            "started_at": datetime.now().isoformat()
        }
        
    async def _stop_scheduler(self) -> Dict[str, Any]:
        """Stop the scheduler"""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
            
        return {
            "status": "stopped",
            "stopped_at": datetime.now().isoformat()
        }
        
    def _run_scheduler(self):
        """Run the scheduler loop"""
        logger.info("Starting monitoring scheduler...")
        while self.scheduler_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    def _scheduled_monitoring_job(self, duration: int):
        """Execute scheduled monitoring job"""
        logger.info(f"Starting scheduled monitoring at {datetime.now()}")
        # This would trigger the detection agent
        # In a real system, this would send a message to the coordinator

# Data Management Agent - Handles data storage and retrieval
class DataAgent(BaseAgent):
    def __init__(self):
        super().__init__("data_agent", "Data Agent")
        self.capabilities = ["data_storage", "result_retrieval", "analytics"]
        self.results_dir = Path("monitoring_results")
        self.results_dir.mkdir(exist_ok=True)
        
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data task"""
        self.status = AgentStatus.BUSY
        
        try:
            task_type = task.get("type")
            
            if task_type == "save_result":
                return await self._save_result(task.get("result"))
            elif task_type == "get_results":
                return await self._get_results(task.get("limit", 10))
            elif task_type == "get_analytics":
                return await self._get_analytics()
            elif task_type == "export_data":
                return await self._export_data()
            else:
                return {"status": "error", "error": f"Unknown task type: {task_type}"}
                
        except Exception as e:
            logger.error(f"Data task failed: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            self.status = AgentStatus.IDLE
            
    async def _save_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Save monitoring result"""
        try:
            # Add metadata
            result["saved_at"] = datetime.now().isoformat()
            result["saved_by"] = self.agent_id
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"monitoring_result_{timestamp}.json"
            filepath = self.results_dir / filename
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
                
            logger.info(f"Result saved to {filename}")
            
            return {
                "status": "saved",
                "filename": filename,
                "filepath": str(filepath)
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
            
    async def _get_results(self, limit: int = 10) -> Dict[str, Any]:
        """Get recent monitoring results"""
        try:
            result_files = list(self.results_dir.glob("monitoring_result_*.json"))
            result_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
            
            results = []
            for file_path in result_files[:limit]:
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
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
            
    async def _get_analytics(self) -> Dict[str, Any]:
        """Generate analytics from stored results"""
        try:
            results_data = await self._get_results(limit=100)  # Get more for analytics
            results = results_data.get("results", [])
            
            if not results:
                return {
                    "total_sessions": 0,
                    "successful_detections": 0,
                    "success_rate": 0,
                    "recent_sessions": 0
                }
            
            total_sessions = len(results)
            successful_detections = sum(1 for r in results if r.get("medicine_intake_detected", False))
            success_rate = (successful_detections / total_sessions * 100) if total_sessions > 0 else 0
            
            # Recent sessions (today)
            today = datetime.now().date()
            recent_sessions = sum(1 for r in results 
                                if datetime.fromisoformat(r.get("monitoring_date", "").replace('Z', '+00:00')).date() == today)
            
            return {
                "total_sessions": total_sessions,
                "successful_detections": successful_detections,
                "success_rate": round(success_rate, 1),
                "recent_sessions": recent_sessions,
                "analytics_generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
            
    async def _export_data(self) -> Dict[str, Any]:
        """Export all data"""
        try:
            results_data = await self._get_results(limit=1000)
            
            # Generate export filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_filename = f"medicine_monitoring_export_{timestamp}.json"
            export_path = self.results_dir / export_filename
            
            # Save export
            with open(export_path, 'w') as f:
                json.dump(results_data, f, indent=2)
                
            return {
                "status": "exported",
                "filename": export_filename,
                "filepath": str(export_path),
                "exported_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Coordinator Agent - Orchestrates all other agents
class CoordinatorAgent(BaseAgent):
    def __init__(self):
        super().__init__("coordinator", "Coordinator Agent")
        self.capabilities = ["workflow_orchestration", "agent_coordination", "task_distribution"]
        self.agents = {}
        self.active_tasks = {}
        
    async def start(self):
        """Start coordinator and all sub-agents"""
        await super().start()
        
        # Initialize and start all agents
        self.agents = {
            "detection": DetectionAgent(),
            "analysis": AnalysisAgent(),
            "scheduling": SchedulingAgent(),
            "data": DataAgent()
        }
        
        # Start all agents
        for agent in self.agents.values():
            await agent.start()
            
        logger.info("All agents started successfully")
        
    async def stop(self):
        """Stop all agents"""
        for agent in self.agents.values():
            await agent.stop()
        await super().stop()
        
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute coordinated task"""
        self.status = AgentStatus.BUSY
        
        try:
            task_type = task.get("type")
            
            if task_type == "analyze_video":
                return await self._coordinate_video_analysis(task)
            elif task_type == "start_live_monitoring":
                return await self._coordinate_live_monitoring(task)
            elif task_type == "manage_schedule":
                return await self._coordinate_scheduling(task)
            elif task_type == "get_analytics":
                return await self._coordinate_analytics(task)
            else:
                return {"status": "error", "error": f"Unknown task type: {task_type}"}
                
        except Exception as e:
            logger.error(f"Coordination task failed: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            self.status = AgentStatus.IDLE
            
    async def _coordinate_video_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate full video analysis workflow"""
        try:
            video_path = task.get("video_path")
            
            # Step 1: Detection
            detection_task = {"type": "analyze_video", "video_path": video_path}
            detection_result = await self.agents["detection"].execute_task(detection_task)
            
            # Step 2: Analysis
            analysis_task = {"type": "analyze_detection", "detection_result": detection_result}
            analysis_result = await self.agents["analysis"].execute_task(analysis_task)
            
            # Step 3: Generate explanation
            explanation_task = {"type": "generate_explanation", "detection_result": detection_result}
            explanation_result = await self.agents["analysis"].execute_task(explanation_task)
            
            # Step 4: Save results
            combined_result = {
                "detection": detection_result,
                "analysis": analysis_result,
                "explanation": explanation_result.get("explanation"),
                "monitoring_date": datetime.now().isoformat(),
                "medicine_intake_detected": detection_result.get("status") == "taken",
                "workflow_completed_by": self.agent_id
            }
            
            save_task = {"type": "save_result", "result": combined_result}
            save_result = await self.agents["data"].execute_task(save_task)
            
            # Step 5: Generate reminder
            reminder = self._generate_reminder(analysis_result)
            
            return {
                "detection": detection_result,
                "analysis": analysis_result,
                "explanation": explanation_result.get("explanation"),
                "reminder": reminder,
                "workflow_status": "completed",
                "save_status": save_result
            }
            
        except Exception as e:
            logger.error(f"Video analysis coordination failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "workflow_status": "failed"
            }
            
    async def _coordinate_live_monitoring(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate live monitoring"""
        try:
            duration = task.get("duration", 60)
            
            # Start monitoring
            monitoring_task = {"type": "start_live_monitoring", "duration": duration}
            result = await self.agents["detection"].execute_task(monitoring_task)
            
            return result
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
            
    async def _coordinate_scheduling(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate scheduling operations"""
        try:
            return await self.agents["scheduling"].execute_task(task)
        except Exception as e:
            return {"status": "error", "error": str(e)}
            
    async def _coordinate_analytics(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate analytics generation"""
        try:
            return await self.agents["data"].execute_task(task)
        except Exception as e:
            return {"status": "error", "error": str(e)}
            
    def _generate_reminder(self, analysis_result: Dict[str, Any]) -> str:
        """Generate reminder message"""
        status = analysis_result.get("status")
        
        if status == "taken":
            return "✅ Excellent! Medicine intake confirmed. Keep maintaining your medication schedule!"
        elif status == "error":
            return "❌ Unable to analyze the recording. Please try again with a clear video or check your camera."
        else:  # missed
            return "⚠️ Medicine intake not detected. Please ensure you take your medication as prescribed and record it properly for tracking."
            
    async def get_system_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {
            "coordinator": {
                "status": self.status.value,
                "active_tasks": len(self.active_tasks)
            }
        }
        
        for agent_name, agent in self.agents.items():
            status[agent_name] = {
                "status": agent.status.value,
                "capabilities": agent.capabilities
            }
            
        return status

# Main AI Agents System
class AIAgentsSystem:
    def __init__(self):
        self.coordinator = CoordinatorAgent()
        self.running = False
        
    async def start(self):
        """Start the AI agents system"""
        await self.coordinator.start()
        self.running = True
        logger.info("AI Agents System started successfully")
        
    async def stop(self):
        """Stop the AI agents system"""
        await self.coordinator.stop()
        self.running = False
        logger.info("AI Agents System stopped")
        
    async def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process video using AI agents"""
        if not self.running:
            return {"status": "error", "error": "System not running"}
            
        task = {"type": "analyze_video", "video_path": video_path}
        return await self.coordinator.execute_task(task)
        
    async def start_live_monitoring(self, duration: int = 60) -> Dict[str, Any]:
        """Start live monitoring using AI agents"""
        if not self.running:
            return {"status": "error", "error": "System not running"}
            
        task = {"type": "start_live_monitoring", "duration": duration}
        return await self.coordinator.execute_task(task)
        
    async def manage_schedule(self, action: str, **kwargs) -> Dict[str, Any]:
        """Manage scheduling using AI agents"""
        if not self.running:
            return {"status": "error", "error": "System not running"}
            
        task = {"type": "manage_schedule", "action": action, **kwargs}
        return await self.coordinator.execute_task(task)
        
    async def get_analytics(self) -> Dict[str, Any]:
        """Get analytics using AI agents"""
        if not self.running:
            return {"status": "error", "error": "System not running"}
            
        task = {"type": "get_analytics"}
        return await self.coordinator.execute_task(task)
        
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        if not self.running:
            return {"status": "offline", "message": "System not running"}
            
        return await self.coordinator.get_system_status()

# Utility functions for backward compatibility
async def multi_agent_workflow(video_path: str) -> Dict[str, Any]:
    """
    Main workflow function for backward compatibility
    """
    system = AIAgentsSystem()
    try:
        await system.start()
        result = await system.process_video(video_path)
        return result
    except Exception as e:
        logger.error(f"Multi-agent workflow failed: {e}")
        return {
            "detection": {"status": "error", "error": str(e)},
            "analysis": {"status": "error", "message": f"Workflow failed: {str(e)}"},
            "explanation": f"An error occurred during video analysis: {str(e)}",
            "reminder": "❌ Unable to process video. Please try again.",
            "workflow_status": "failed"
        }
    finally:
        await system.stop()

# Example usage and testing
if __name__ == "__main__":
    import sys
    import asyncio
    
    async def main():
        if len(sys.argv) > 1:
            if sys.argv[1] == "--test":
                # Test the system
                system = AIAgentsSystem()
                await system.start()
                
                # Test system status
                status = await system.get_system_status()
                print("System Status:")
                print(json.dumps(status, indent=2))
                
                # Test with mock video analysis
                result = await system.process_video("test_video.mp4")
                print("\nVideo Analysis Result:")
                print(json.dumps(result, indent=2))
                
                await system.stop()
                
            elif sys.argv[1] == "--live":
                # Test live monitoring
                duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
                system = AIAgentsSystem()
                await system.start()
                
                result = await system.start_live_monitoring(duration)
                print(f"Live monitoring result: {json.dumps(result, indent=2)}")
                
                await system.stop()
                
            else:
                # Process video file
                result = await multi_agent_workflow(sys.argv[1])
                print(json.dumps(result, indent=2))
        else:
            print("Usage:")
            print("  python ai_agents_system.py --test")
            print("  python ai_agents_system.py --live [duration_seconds]")
            print("  python ai_agents_system.py video.mp4")
    
    asyncio.run(main())
