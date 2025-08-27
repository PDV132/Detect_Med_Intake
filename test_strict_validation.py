"""
Test Script for Strict Medicine Detection with Three-Condition Validation
Demonstrates the enhanced AI agents system with strict validation requirements:
1. One hand holding medicine strip/bottle
2. Other hand with fingers near mouth
3. Head tilted back for taking medicine
"""

import asyncio
import json
import logging
from pathlib import Path
import sys
import time
from ai_agents_system import AIAgentsSystem
from enhanced_medicine_detection import StrictMedicineDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")

def print_condition_status(condition_name: str, status: bool, details: str = ""):
    """Print condition validation status"""
    icon = "✅" if status else "❌"
    print(f"  {icon} {condition_name}: {'PASSED' if status else 'FAILED'}")
    if details:
        print(f"     {details}")

async def test_ai_agents_system():
    """Test the AI agents system with strict validation"""
    print_header("🤖 AI AGENTS SYSTEM - STRICT VALIDATION TEST")
    
    system = AIAgentsSystem()
    
    try:
        # Start the system
        print("🚀 Starting AI Agents System...")
        await system.start()
        print("✅ System started successfully")
        
        # Test system status
        print_section("📊 System Status Check")
        status = await system.get_system_status()
        
        print("Agent Status:")
        for agent_name, agent_info in status.items():
            agent_status = agent_info.get('status', 'unknown')
            capabilities = agent_info.get('capabilities', [])
            
            status_icon = "🟢" if agent_status == "idle" else "🔴" if agent_status == "error" else "🟡"
            print(f"  {status_icon} {agent_name.title()}: {agent_status}")
            
            if capabilities:
                print(f"     Capabilities: {', '.join(capabilities)}")
        
        # Test with available video files
        video_files = [
            "med_intake.mp4",
            "intake_med_1.mp4", 
            "med_intake1.mp4"
        ]
        
        found_videos = [f for f in video_files if Path(f).exists()]
        
        if not found_videos:
            print_section("⚠️  No Video Files Found")
            print("Available video files not found. Testing with mock data...")
            
            # Create a mock test
            await test_mock_validation(system)
        else:
            print_section(f"🎬 Testing with {len(found_videos)} Video Files")
            
            for video_file in found_videos:
                await test_video_analysis(system, video_file)
        
        # Test analytics
        print_section("📈 Analytics Test")
        analytics = await system.get_analytics()
        
        print("System Analytics:")
        print(f"  📊 Total Sessions: {analytics.get('total_sessions', 0)}")
        print(f"  ✅ Successful Detections: {analytics.get('successful_detections', 0)}")
        print(f"  📈 Success Rate: {analytics.get('success_rate', 0)}%")
        print(f"  📅 Recent Sessions: {analytics.get('recent_sessions', 0)}")
        
    except Exception as e:
        logger.error(f"System test failed: {e}")
        print(f"❌ Test failed: {e}")
        
    finally:
        print_section("🛑 Shutting Down System")
        await system.stop()
        print("✅ System stopped successfully")

async def test_video_analysis(system: AIAgentsSystem, video_file: str):
    """Test video analysis with detailed condition checking"""
    print_section(f"🎥 Analyzing: {video_file}")
    
    start_time = time.time()
    result = await system.process_video(video_file)
    analysis_time = time.time() - start_time
    
    print(f"⏱️  Analysis completed in {analysis_time:.2f} seconds")
    
    # Extract results
    detection = result.get("detection", {})
    analysis = result.get("analysis", {})
    explanation = result.get("explanation", "")
    
    # Show basic results
    status = detection.get("status", "unknown")
    validation_method = detection.get("validation_method", "standard")
    
    print(f"\n📋 Results Summary:")
    print(f"  Status: {status.upper()}")
    print(f"  Method: {validation_method}")
    print(f"  Detection Agent: {detection.get('detection_method', 'unknown')}")
    
    if status == "taken":
        print("🎉 MEDICINE INTAKE VALIDATED!")
        
        events = detection.get("events", [])
        if events and validation_method == "strict_three_condition_check":
            print_section("🔍 Strict Validation Details")
            
            event = events[0]
            validation_details = event.get("validation_details", {})
            
            # Check each condition
            print("Three-Condition Validation Results:")
            
            # Condition 1: Medicine in hand
            condition_1 = validation_details.get("condition_1_medicine_in_hand", False)
            med_details = validation_details.get("medicine_object_details", {})
            if condition_1 and med_details:
                object_type = med_details.get("object_type", "unknown")
                confidence = med_details.get("confidence", 0)
                hand = med_details.get("hand_holding", "unknown")
                details = f"{object_type.replace('_', ' ').title()} ({confidence:.0%} confidence, {hand} hand)"
            else:
                details = "No medicine object detected"
            print_condition_status("Medicine Object in Hand", condition_1, details)
            
            # Condition 2: Hand near mouth
            condition_2 = validation_details.get("condition_2_hand_near_mouth", False)
            hand_details = validation_details.get("hand_mouth_details", {})
            if condition_2 and hand_details:
                finger = hand_details.get("closest_finger", "unknown")
                distance = hand_details.get("distance", 0)
                hand_label = hand_details.get("hand_label", "unknown")
                details = f"{finger.title()} finger (distance: {distance:.3f}, {hand_label} hand)"
            else:
                details = "Hand not positioned near mouth"
            print_condition_status("Hand Near Mouth", condition_2, details)
            
            # Condition 3: Head tilted back
            condition_3 = validation_details.get("condition_3_head_tilted_back", False)
            head_details = validation_details.get("head_tilt_details", {})
            if condition_3 and head_details:
                angle = head_details.get("tilt_angle_estimate", 0)
                ratio = head_details.get("tilt_ratio", 0)
                details = f"Tilt angle: ~{angle:.1f}° (ratio: {ratio:.3f})"
            else:
                details = "Head not tilted back sufficiently"
            print_condition_status("Head Tilted Back", condition_3, details)
            
            # Overall confidence
            overall_confidence = validation_details.get("overall_confidence", 0)
            print(f"\n🎯 Overall Confidence: {overall_confidence:.0%}")
            
            # Show confidence breakdown
            confidence_scores = validation_details.get("confidence_scores", {})
            if confidence_scores:
                print("\n📊 Confidence Breakdown:")
                for metric, score in confidence_scores.items():
                    print(f"  • {metric.replace('_', ' ').title()}: {score:.0%}")
            
            # Show frame statistics
            total_valid_frames = detection.get("total_valid_frames", 0)
            total_frames = detection.get("total_frames", 0)
            print(f"\n📈 Frame Statistics:")
            print(f"  • Valid frames: {total_valid_frames}")
            print(f"  • Total frames: {total_frames}")
            if total_frames > 0:
                print(f"  • Validation rate: {(total_valid_frames/total_frames)*100:.1f}%")
    
    elif status == "missed":
        print("❌ MEDICINE INTAKE NOT VALIDATED")
        
        if validation_method == "strict_three_condition_check":
            failure_details = detection.get("failure_details", [])
            print("\n❗ Validation Failures:")
            for detail in failure_details:
                print(f"  • {detail}")
            
            conditions_required = detection.get("conditions_required", [])
            print("\n📋 Required Conditions:")
            for condition in conditions_required:
                print(f"  • {condition}")
    
    elif status == "error":
        print("💥 DETECTION ERROR")
        error_msg = detection.get("error", "Unknown error")
        print(f"  Error: {error_msg}")
    
    # Show AI explanation
    if explanation:
        print_section("🤖 AI Analysis & Explanation")
        print(explanation)
    
    # Show reminder
    reminder = result.get("reminder", "")
    if reminder:
        print_section("🔔 Reminder")
        print(reminder)
    
    print("\n" + "="*80)

async def test_mock_validation(system: AIAgentsSystem):
    """Test with mock validation data"""
    print_section("🧪 Mock Validation Test")
    
    print("Creating mock video analysis...")
    
    # This would typically process a real video, but for demo purposes
    # we'll show what the system would detect
    
    mock_result = {
        "detection": {
            "status": "taken",
            "validation_method": "strict_three_condition_check",
            "events": [{
                "timestamp_sec": 3.5,
                "validation_details": {
                    "condition_1_medicine_in_hand": True,
                    "condition_2_hand_near_mouth": True,
                    "condition_3_head_tilted_back": True,
                    "all_conditions_met": True,
                    "medicine_object_details": {
                        "medicine_object_detected": True,
                        "object_type": "medicine_strip",
                        "confidence": 0.85,
                        "hand_holding": "Left"
                    },
                    "hand_mouth_details": {
                        "distance": 0.08,
                        "closest_finger": "index",
                        "hand_label": "Right"
                    },
                    "head_tilt_details": {
                        "tilt_angle_estimate": 15.0,
                        "tilt_ratio": -0.12
                    },
                    "confidence_scores": {
                        "medicine_object": 0.85,
                        "hand_mouth_distance": 0.78,
                        "head_tilt": 0.82
                    },
                    "overall_confidence": 0.82
                }
            }],
            "total_valid_frames": 25,
            "total_frames": 150,
            "detection_method": "strict_validation"
        },
        "analysis": {
            "status": "taken",
            "confidence": "high"
        },
        "explanation": "✅ Medicine intake successfully validated using strict three-condition check: Medicine strip detected in left hand, index finger positioned near mouth, head tilted back (~15° angle). Total valid frames: 25",
        "reminder": "✅ Excellent! Medicine intake confirmed. Keep maintaining your medication schedule!"
    }
    
    print("📊 Mock Results:")
    print(f"  Status: {mock_result['detection']['status'].upper()}")
    print(f"  Method: {mock_result['detection']['validation_method']}")
    print(f"  Valid Frames: {mock_result['detection']['total_valid_frames']}")
    
    print("\n🎯 Mock Validation Details:")
    validation_details = mock_result['detection']['events'][0]['validation_details']
    
    print_condition_status("Medicine Strip in Hand", 
                          validation_details['condition_1_medicine_in_hand'],
                          "Medicine strip (85% confidence, Left hand)")
    
    print_condition_status("Hand Near Mouth", 
                          validation_details['condition_2_hand_near_mouth'],
                          "Index finger (distance: 0.080, Right hand)")
    
    print_condition_status("Head Tilted Back", 
                          validation_details['condition_3_head_tilted_back'],
                          "Tilt angle: ~15.0° (ratio: -0.120)")
    
    print(f"\n🎯 Overall Confidence: {validation_details['overall_confidence']:.0%}")

async def test_direct_detector():
    """Test the direct strict detector"""
    print_header("🔬 DIRECT STRICT DETECTOR TEST")
    
    detector = StrictMedicineDetector()
    
    try:
        video_files = [
            "med_intake.mp4",
            "intake_med_1.mp4", 
            "med_intake1.mp4"
        ]
        
        found_videos = [f for f in video_files if Path(f).exists()]
        
        if not found_videos:
            print("⚠️  No video files found for direct testing")
            print("Available test files: med_intake.mp4, intake_med_1.mp4, med_intake1.mp4")
        else:
            for video_file in found_videos:
                print_section(f"📹 Direct Analysis: {video_file}")
                
                start_time = time.time()
                result = detector.process_video_with_strict_validation(video_file)
                analysis_time = time.time() - start_time
                
                print(f"⏱️  Analysis time: {analysis_time:.2f} seconds")
                print(f"📊 Status: {result.get('status', 'unknown').upper()}")
                print(f"🔍 Method: {result.get('validation_method', 'unknown')}")
                print(f"📈 Valid frames: {result.get('total_valid_frames', 0)}")
                print(f"📊 Total frames: {result.get('total_frames', 0)}")
                
                if result.get("status") == "taken":
                    print("✅ Direct validation PASSED")
                    conditions = result.get("conditions_validated", [])
                    print("✅ Validated conditions:")
                    for condition in conditions:
                        print(f"  • {condition}")
                else:
                    print("❌ Direct validation FAILED")
                    failure_details = result.get("failure_details", [])
                    print("❌ Failure reasons:")
                    for detail in failure_details:
                        print(f"  • {detail}")
                
                print()
    
    finally:
        detector.release()

def show_validation_requirements():
    """Show the three validation requirements"""
    print_header("📚 STRICT THREE-CONDITION VALIDATION REQUIREMENTS")
    
    print("""
🎯 CONDITION 1: Medicine Object in Hand
   • One hand must be holding a medicine strip, bottle, or package
   • Detects medicine strips (blister packs) with metallic backing
   • Identifies medicine bottles (plastic/glass containers)  
   • Recognizes medicine packages with rectangular shapes
   • Uses color analysis, shape detection, and edge detection
   • Requires 70%+ confidence for validation

🎯 CONDITION 2: Hand Near Mouth  
   • The other hand must have fingers positioned near the mouth
   • Analyzes all five finger tips (thumb, index, middle, ring, pinky)
   • Calculates distance from closest finger to mouth area
   • Mouth area determined from face detection landmarks
   • Distance threshold: 0.12 (normalized coordinates)
   • Must be the hand NOT holding the medicine

🎯 CONDITION 3: Head Tilted Back
   • Head must be tilted backward as when swallowing medicine
   • Uses pose landmarks to detect head orientation
   • Compares nose position relative to ear positions  
   • Nose should be higher than ears when head tilted back
   • Tilt threshold: -0.08 (normalized ratio)
   • Estimates tilt angle for user feedback

✅ VALIDATION SUCCESS CRITERIA:
   • ALL three conditions must be met simultaneously
   • Must be detected for at least 10 consecutive frames
   • Each condition has individual confidence scoring
   • Overall confidence calculated from all conditions
   • Detailed feedback provided for failed validations

💡 TIPS FOR SUCCESSFUL DETECTION:
   • Hold medicine strip/bottle clearly visible in one hand
   • Bring other hand close to mouth (fingers near lips)
   • Tilt head back slightly when swallowing
   • Ensure good lighting and clear camera view
   • Keep upper body visible to camera
   • Maintain steady position during recording
    """)

async def main():
    """Main test function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--requirements":
            show_validation_requirements()
            return
        elif sys.argv[1] == "--direct":
            await test_direct_detector()
            return
        elif sys.argv[1] == "--agents":
            await test_ai_agents_system()
            return
    
    # Run all tests by default
    show_validation_requirements()
    await test_ai_agents_system()
    await test_direct_detector()

if __name__ == "__main__":
    print("🧪 STRICT MEDICINE DETECTION VALIDATION TEST SUITE")
    print("Testing three-condition validation system...")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        logger.error(f"Test suite error: {e}")
    
    print("\n✅ Test suite completed")
