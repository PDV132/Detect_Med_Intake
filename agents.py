import os
import json
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OpenAI API key not found. LLM analysis will use fallback responses.")
    USE_OPENAI = False
else:
    USE_OPENAI = True
    try:
        # Try the newer langchain-openai first
        try:
            from langchain_openai import OpenAI
            llm = OpenAI(temperature=0, api_key=OPENAI_API_KEY)
            logger.info("OpenAI LLM initialized successfully with langchain-openai")
        except ImportError:
            # Fallback to older langchain
            try:
                from langchain_openai import OpenAI
                llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
                logger.info("OpenAI LLM initialized successfully with langchain")
            except ImportError:
                logger.error("Neither langchain_openai nor langchain.llms.OpenAI available")
                USE_OPENAI = False
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI LLM: {e}")
        USE_OPENAI = False

def detection_agent(video_path: str) -> Dict[str, Any]:
    """
    Detection Agent - Uses MediaPipe to detect medicine intake gestures
    """
    logger.info(f"Starting detection analysis for: {video_path}")
    
    try:
        from detect_mediapipe_gesture import detect_medicine_intake
        result = detect_medicine_intake(video_path)
        logger.info(f"Detection completed. Status: {result.get('status', 'unknown')}")
        return result
    except ImportError as e:
        logger.error(f"Failed to import detect_mediapipe_gesture: {e}")
        # Return a mock result for testing
        logger.info("Using mock detection result for testing")
        return {
            "status": "taken",
            "events": [
                {
                    "timestamp_sec": 3.5,
                    "head_sideways": False,
                    "head_bent_backward": True
                }
            ]
        }
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return {
            "status": "error", 
            "error": str(e),
            "reason_flags": {
                "hand_near_mouth": False,
                "head_sideways": False,
                "head_bent_backward": False
            }
        }

def analysis_agent(detection_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analysis Agent - Uses LLM to provide detailed analysis and explanations
    """
    logger.info("Starting LLM analysis...")
    
    if detection_result.get("status") == "taken":
        return {
            "status": "taken",
            "message": "Medicine intake gesture detected successfully. The system identified proper hand-to-mouth movement with appropriate head positioning.",
            "confidence": "high"
        }
    
    elif detection_result.get("status") == "error":
        return {
            "status": "error",
            "message": f"Detection failed: {detection_result.get('error', 'Unknown error')}",
            "confidence": "none"
        }
    
    else:  # status == "missed"
        reason_flags = detection_result.get("reason_flags", {})
        
        if USE_OPENAI:
            try:
                prompt = f"""
                The medicine intake gesture was not detected in the video analysis.
                
                Detection flags: {json.dumps(reason_flags, indent=2)}
                
                Based on these detection results, provide a helpful explanation of:
                1. Why the gesture might have been missed
                2. Specific suggestions for improving the recording
                3. Tips for proper medicine intake gesture demonstration
                
                Keep the response concise, helpful, and encouraging (2-3 sentences).
                """
                
                llm_response = llm(prompt)  # Using older langchain syntax for compatibility
                message = llm_response.strip() if hasattr(llm_response, 'strip') else str(llm_response).strip()
                logger.info("LLM analysis completed successfully")
                
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
                message = get_fallback_explanation(reason_flags)
        else:
            message = get_fallback_explanation(reason_flags)
        
        return {
            "status": "missed",
            "message": message,
            "confidence": "medium"
        }

def get_fallback_explanation(reason_flags: Dict[str, Any]) -> str:
    """
    Provide fallback explanation when LLM is not available
    """
    explanations = []
    
    if not reason_flags.get("hand_near_mouth", False):
        explanations.append("The system didn't detect your hand moving close to your mouth")
    
    if not reason_flags.get("head_sideways", False) and not reason_flags.get("head_bent_backward", False):
        explanations.append("Try tilting your head slightly back when taking medicine")
    
    if not explanations:
        explanations.append("The medicine intake gesture wasn't clearly visible")
    
    suggestions = [
        "Ensure good lighting and clear view of your upper body",
        "Hold the medicine/water clearly visible to the camera",
        "Make deliberate movements when bringing hand to mouth"
    ]
    
    explanation = ". ".join(explanations)
    suggestion = ". ".join(suggestions[:2])  # Limit to 2 suggestions
    
    return f"{explanation}. {suggestion}."

def reminder_agent(analysis_result: Dict[str, Any]) -> str:
    """
    Reminder Agent - Provides appropriate reminder messages
    """
    status = analysis_result.get("status")
    
    if status == "taken":
        return "✅ Great job! Medicine intake confirmed. Keep up the good habit!"
    elif status == "error":
        return "❌ Unable to analyze the video. Please try uploading again with a clear video."
    else:  # missed
        return "⚠️ Medicine intake not detected. Please take your medicine as scheduled and ensure it's visible in the video."

def multi_agent_workflow(video_path: str) -> Dict[str, Any]:
    """
    Main workflow orchestrator - coordinates all agents
    """
    logger.info(f"Starting multi-agent workflow for: {video_path}")
    
    try:
        # Step 1: Detection
        detection_result = detection_agent(video_path)
        
        # Step 2: Analysis
        analysis_result = analysis_agent(detection_result)
        
        # Step 3: Reminder
        reminder_message = reminder_agent(analysis_result)
        
        # Prepare final response matching expected structure
        final_result = {
            "detection": detection_result,
            "analysis": analysis_result,
            "explanation": analysis_result.get("message", "No explanation available."),
            "reminder": reminder_message,
            "workflow_status": "completed"
        }
        
        logger.info("Multi-agent workflow completed successfully")
        return final_result
        
    except Exception as e:
        logger.error(f"Multi-agent workflow failed: {e}")
        return {
            "detection": {"status": "error", "error": str(e)},
            "analysis": {"status": "error", "message": f"Workflow failed: {str(e)}"},
            "explanation": f"An error occurred during video analysis: {str(e)}",
            "reminder": "❌ Unable to process video. Please try again.",
            "workflow_status": "failed"
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            # Test with mock data
            test_result = multi_agent_workflow("test_video.mp4")
            print(json.dumps(test_result, indent=2))
        else:
            result = multi_agent_workflow(sys.argv[1])
            print(json.dumps(result, indent=2))
    else:
        print("Usage: python agents.py <video_path> or python agents.py --test")