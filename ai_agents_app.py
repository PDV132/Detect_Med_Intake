"""
AI Agents Streamlit App for Medicine Intake Detection
Enhanced Streamlit frontend that uses AI agents for all functionality
"""

import streamlit as st
import requests
import time
import json
import asyncio
from datetime import datetime, timedelta
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_URL = "http://localhost:8000"
ENDPOINTS = {
    "upload": f"{API_URL}/upload-video/",
    "test": f"{API_URL}/test-detection/",
    "health": f"{API_URL}/health",
    "agents_status": f"{API_URL}/agents-status/",
    "start_monitoring": f"{API_URL}/start-live-monitoring/",
    "stop_monitoring": f"{API_URL}/stop-monitoring/",
    "monitoring_status": f"{API_URL}/monitoring-status/",
    "schedule": f"{API_URL}/schedule-monitoring/",
    "scheduled_times": f"{API_URL}/scheduled-times/",
    "clear_schedule": f"{API_URL}/clear-schedule/",
    "monitoring_results": f"{API_URL}/monitoring-results/",
    "analytics": f"{API_URL}/analytics/",
    "export_results": f"{API_URL}/export-results/"
}

# Page configuration
st.set_page_config(
    page_title="AI Agents Medicine Intake Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_api_health():
    """Check if the AI agents API backend is running"""
    try:
        response = requests.get(ENDPOINTS["health"], timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return False, None

def get_agents_status():
    """Get detailed AI agents status"""
    try:
        response = requests.get(ENDPOINTS["agents_status"], timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Failed to get agents status: {e}")
    return None

def get_monitoring_status():
    """Get current monitoring status from AI agents"""
    try:
        response = requests.get(ENDPOINTS["monitoring_status"], timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Failed to get monitoring status: {e}")
    return None

def start_live_monitoring(duration: int):
    """Start live monitoring using AI agents"""
    try:
        payload = {"duration": duration}
        response = requests.post(ENDPOINTS["start_monitoring"], json=payload, timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def stop_monitoring():
    """Stop active monitoring using AI agents"""
    try:
        response = requests.post(ENDPOINTS["stop_monitoring"], timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def schedule_monitoring(time_str: str, duration: int):
    """Schedule daily monitoring using AI agents"""
    try:
        payload = {"time": time_str, "duration": duration}
        response = requests.post(ENDPOINTS["schedule"], json=payload, timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def get_monitoring_results():
    """Get monitoring results from AI agents"""
    try:
        response = requests.get(ENDPOINTS["monitoring_results"], timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Failed to get monitoring results: {e}")
    return None

def get_analytics():
    """Get analytics from AI agents"""
    try:
        response = requests.get(ENDPOINTS["analytics"], timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Failed to get analytics: {e}")
    return None

def display_agents_status():
    """Display AI agents status in sidebar"""
    st.sidebar.subheader("🤖 AI Agents Status")
    
    agents_status = get_agents_status()
    if agents_status:
        system_running = agents_status.get("system_running", False)
        
        if system_running:
            st.sidebar.success("✅ AI Agents System: Running")
        else:
            st.sidebar.error("❌ AI Agents System: Offline")
        
        # Display individual agent status
        agents = agents_status.get("agents", {})
        for agent_name, agent_info in agents.items():
            status = agent_info.get("status", "unknown")
            if status == "idle":
                st.sidebar.info(f"🟢 {agent_name.title()} Agent: Ready")
            elif status == "busy":
                st.sidebar.warning(f"🟡 {agent_name.title()} Agent: Working")
            else:
                st.sidebar.error(f"🔴 {agent_name.title()} Agent: {status.title()}")
    else:
        st.sidebar.error("❌ Cannot connect to AI Agents System")

def display_result_summary(result):
    """Display a summary of the AI agents detection results"""
    detection = result.get("detection", {})
    analysis = result.get("analysis", {})
    
    status = detection.get("status", "unknown")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if status == "taken":
            st.success("🎉 Medicine Intake Detected by AI!")
            events = detection.get("events", [])
            if events:
                st.info(f"Detected at {events[0].get('timestamp_sec', 0):.1f} seconds")
        elif status == "missed":
            st.warning("⚠️ Medicine Intake Not Detected")
        elif status == "error":
            st.error("❌ AI Detection Error")
        else:
            st.info("ℹ️ Unknown Status")
    
    with col2:
        # Medicine object detection
        events = detection.get("events", [])
        if events and events[0].get("detection_details", {}).get("medicine_object_detected"):
            object_type = events[0]["detection_details"].get("object_type", "unknown")
            st.info(f"🏥 {object_type.replace('_', ' ').title()} Detected")
        else:
            st.warning("🏥 No Medicine Object Detected")
    
    with col3:
        # AI Analysis confidence
        confidence = analysis.get("confidence", "unknown")
        analyzed_by = analysis.get("analyzed_by", "AI Agent")
        if confidence == "high":
            st.success(f"📊 AI Confidence: {confidence.title()}")
        elif confidence == "medium":
            st.warning(f"📊 AI Confidence: {confidence.title()}")
        else:
            st.error(f"📊 AI Confidence: {confidence.title()}")
        
        st.caption(f"Analyzed by: {analyzed_by}")

def display_enhanced_results(result):
    """Display enhanced results with AI agents processing info"""
    
    # AI Agents Processing Results
    with st.expander("🤖 AI Agents Detection Results", expanded=True):
        detection = result.get("detection", {})
        
        # Basic info with AI processing details
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status", detection.get("status", "unknown").title())
        with col2:
            st.metric("Processing Time", f"{detection.get('processing_time', 0):.1f}s")
        with col3:
            st.metric("File Size", f"{detection.get('file_size_mb', 0):.1f}MB")
        with col4:
            processed_by = detection.get("processed_by", "AI Agent")
            st.metric("Processed By", processed_by.replace("_", " ").title())
        
        # Event details with AI analysis
        events = detection.get("events", [])
        if events:
            st.write("**AI Detection Events:**")
            for i, event in enumerate(events):
                st.write(f"**Event {i+1} (AI Analyzed):**")
                
                details = event.get("detection_details", {})
                
                # Create metrics for detection details
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Timestamp", f"{event.get('timestamp_sec', 0):.1f}s")
                with col2:
                    st.metric("Hand Near Mouth", "✅" if details.get("hand_near_mouth") else "❌")
                with col3:
                    st.metric("Head Tilted Back", "✅" if details.get("head_tilted_back") else "❌")
                with col4:
                    if details.get("medicine_object_detected"):
                        object_type = details.get("object_type", "unknown")
                        st.metric("Medicine Object", object_type.replace("_", " ").title())
                    else:
                        st.metric("Medicine Object", "Not Detected")
                
                # AI Confidence scores
                confidence_scores = details.get("confidence_scores", {})
                if confidence_scores:
                    st.write("**AI Confidence Scores:**")
                    score_cols = st.columns(len(confidence_scores))
                    for idx, (key, score) in enumerate(confidence_scores.items()):
                        with score_cols[idx]:
                            st.metric(key.replace("_", " ").title(), f"{score:.0%}")

def display_monitoring_dashboard():
    """Display AI agents live monitoring dashboard"""
    st.header("🤖 AI Agents Live Monitoring Dashboard")
    
    # Get current status from AI agents
    status = get_monitoring_status()
    
    if status:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if status.get("ai_system_running"):
                st.success("🤖 AI Agents System: Running")
            else:
                st.error("🤖 AI Agents System: Offline")
        
        with col2:
            agents_status = status.get("agents_status", {})
            detection_status = agents_status.get("detection", {}).get("status", "unknown")
            if detection_status == "busy":
                st.warning("📹 Detection Agent: Monitoring")
            elif detection_status == "idle":
                st.info("📹 Detection Agent: Ready")
            else:
                st.error(f"📹 Detection Agent: {detection_status}")
        
        with col3:
            scheduled_count = len(status.get("scheduled_times", []))
            st.metric("Scheduled Times", scheduled_count)
    
    # AI Agents monitoring controls
    st.subheader("🎬 Start AI Agents Live Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.slider("Monitoring Duration (seconds)", 30, 300, 60)
        
        if st.button("▶️ Start AI Agents Monitoring", type="primary"):
            with st.spinner("Starting AI agents live monitoring..."):
                success, response = start_live_monitoring(duration)
                
                if success:
                    st.success(f"✅ AI agents live monitoring started for {duration} seconds!")
                    if response.get("agents_coordinated"):
                        st.info("🤖 All AI agents coordinated successfully")
                    st.json(response)
                else:
                    st.error(f"❌ Failed to start AI monitoring: {response.get('error', 'Unknown error')}")
    
    with col2:
        if st.button("⏹️ Stop AI Monitoring", type="secondary"):
            success, response = stop_monitoring()
            
            if success:
                st.success("✅ AI agents monitoring stopped!")
                st.info("🤖 AI agents coordination completed")
            else:
                st.error(f"❌ Failed to stop AI monitoring: {response.get('error', 'Unknown error')}")

def display_scheduling_interface():
    """Display AI agents scheduling interface"""
    st.header("⏰ AI Agents Schedule Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Add New AI Schedule")
        
        # Time input
        schedule_time = st.time_input(
            "Monitoring Time",
            value=datetime.strptime("09:00", "%H:%M").time(),
            help="AI agents will automatically start monitoring at this time"
        )
        
        schedule_duration = st.slider(
            "Duration (seconds)", 
            30, 300, 60,
            help="How long AI agents will monitor each day"
        )
        
        if st.button("➕ Add AI Schedule", type="primary"):
            time_str = schedule_time.strftime("%H:%M")
            success, response = schedule_monitoring(time_str, schedule_duration)
            
            if success:
                st.success(f"✅ AI agents monitoring scheduled for {time_str} daily!")
                if response.get("agent_coordinated"):
                    st.info("🤖 Scheduling agent coordinated successfully")
                st.rerun()
            else:
                st.error(f"❌ Failed to schedule AI agents: {response.get('error', 'Unknown error')}")
    
    with col2:
        st.subheader("Current AI Schedule")
        
        # Get scheduled times from AI agents
        try:
            response = requests.get(ENDPOINTS["scheduled_times"], timeout=5)
            if response.status_code == 200:
                schedule_data = response.json()
                scheduled_times = schedule_data.get("scheduled_times", [])
                
                if scheduled_times:
                    st.info("🤖 Managed by AI Scheduling Agent")
                    for schedule in scheduled_times:
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"⏰ {schedule['time']} - {schedule['duration']}s (AI Managed)")
                    
                    if st.button("🗑️ Clear All AI Schedules", type="secondary"):
                        try:
                            clear_response = requests.delete(ENDPOINTS["clear_schedule"], timeout=5)
                            if clear_response.status_code == 200:
                                result = clear_response.json()
                                st.success("✅ All AI schedules cleared!")
                                if result.get("agent_coordinated"):
                                    st.info("🤖 AI agents coordination completed")
                                st.rerun()
                            else:
                                st.error("❌ Failed to clear AI schedules")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                else:
                    st.info("No AI scheduled monitoring times")
            else:
                st.error("Failed to fetch AI schedule")
        except Exception as e:
            st.error(f"Error fetching AI schedule: {e}")

def display_monitoring_results():
    """Display AI agents monitoring results and analytics"""
    st.header("📊 AI Agents Results & Analytics")
    
    results_data = get_monitoring_results()
    
    if not results_data or not results_data.get("results"):
        st.info("No AI monitoring results available yet")
        return
    
    results = results_data["results"]
    
    # AI Processing indicator
    if results_data.get("processed_by_ai_agents"):
        st.success("🤖 Results processed and managed by AI Agents")
    
    # Summary metrics with AI enhancement
    st.subheader("📈 AI-Enhanced Summary")
    
    analytics_data = get_analytics()
    if analytics_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sessions", analytics_data.get("total_sessions", 0))
        with col2:
            st.metric("Successful Detections", analytics_data.get("successful_detections", 0))
        with col3:
            st.metric("AI Success Rate", f"{analytics_data.get('success_rate', 0)}%")
        with col4:
            st.metric("Recent Sessions", analytics_data.get("recent_sessions", 0))
        
        if analytics_data.get("generated_by_ai_agents"):
            st.caption("📊 Analytics generated by AI Data Agent")
    
    # Results timeline with AI processing info
    st.subheader("📅 AI Detection Timeline")
    
    if results:
        # Prepare data for visualization
        timeline_data = []
        for result in results:
            try:
                date_str = result.get("monitoring_date", "")
                if date_str:
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    timeline_data.append({
                        "Date": dt.date(),
                        "Time": dt.time(),
                        "Detected": result.get("medicine_intake_detected", False),
                        "Duration": result.get("duration_seconds", 0),
                        "Events": result.get("total_detections", 0),
                        "AI_Processed": "Yes" if result.get("workflow_completed_by") else "No"
                    })
            except Exception as e:
                logger.warning(f"Error processing result: {e}")
        
        if timeline_data:
            df = pd.DataFrame(timeline_data)
            
            # Create timeline chart with AI processing indicator
            fig = px.scatter(df, 
                            x="Date", 
                            y="Time",
                            color="Detected",
                            size="Events",
                            symbol="AI_Processed",
                            hover_data=["Duration"],
                            title="AI Agents Medicine Intake Detection Timeline")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Success rate over time
            daily_stats = df.groupby("Date").agg({
                "Detected": ["sum", "count"]
            }).round(2)
            
            daily_stats.columns = ["Successful", "Total"]
            daily_stats["Success_Rate"] = (daily_stats["Successful"] / daily_stats["Total"] * 100).round(1)
            
            fig2 = px.line(daily_stats.reset_index(), 
                          x="Date", 
                          y="Success_Rate",
                          title="Daily AI Success Rate (%)",
                          markers=True)
            
            st.plotly_chart(fig2, use_container_width=True)
    
    # Detailed results with AI processing info
    st.subheader("📋 Detailed AI Results")
    
    for i, result in enumerate(results[:10]):  # Show last 10 results
        with st.expander(f"AI Session {i+1}: {result.get('monitoring_date', '')[:19]}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("AI Detection Status", 
                         "✅ Detected" if result.get("medicine_intake_detected") else "❌ Not Detected")
            with col2:
                st.metric("Duration", f"{result.get('duration_seconds', 0)}s")
            with col3:
                st.metric("Events", result.get("total_detections", 0))
            
            # AI processing info
            if result.get("workflow_completed_by"):
                st.info(f"🤖 Processed by: {result['workflow_completed_by'].replace('_', ' ').title()}")
            
            if result.get("detection_events"):
                st.write("**AI Detection Events:**")
                for event in result["detection_events"]:
                    st.json(event)

def main():
    st.title("🤖 AI Agents Medicine Intake Detection System")
    st.markdown("Advanced AI agent-based medicine intake detection with intelligent coordination, analysis, and monitoring.")
    
    # Sidebar with AI agents status
    with st.sidebar:
        st.header("🛠️ System Controls")
        
        # AI Agents Status
        display_agents_status()
        
        # API Health Check
        st.subheader("System Health")
        if st.button("🔄 Check AI System Status"):
            with st.spinner("Checking AI agents system..."):
                is_healthy, health_data = check_api_health()
                if is_healthy:
                    st.success("✅ AI Agents System is running")
                    if health_data:
                        if health_data.get("ai_system_running"):
                            st.success("🤖 AI Agents: Active")
                        st.json(health_data)
                else:
                    st.error("❌ AI Agents System is not accessible")
                    st.error("Make sure to run: `uvicorn ai_agents_api_backend:app --reload`")
        
        # Navigation
        st.subheader("📍 Navigation")
        page = st.selectbox(
            "Select Page",
            ["🎬 Video Upload", "📹 AI Live Monitoring", "⏰ AI Scheduling", "📊 AI Results & Analytics"],
            help="Choose which AI-powered feature to use"
        )
    
    # Main content based on selected page
    if page == "🎬 Video Upload":
        display_video_upload_page()
    elif page == "📹 AI Live Monitoring":
        display_monitoring_dashboard()
    elif page == "⏰ AI Scheduling":
        display_scheduling_interface()
    elif page == "📊 AI Results & Analytics":
        display_monitoring_results()

def display_video_upload_page():
    """Display the AI agents video upload page"""
    st.header("🎬 AI Agents Video Upload & Analysis")
    
    # File uploader
    video_file = st.file_uploader(
        "Choose a video file for AI analysis",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a video showing medicine intake. AI agents will analyze gesture, object detection, and provide intelligent explanations!"
    )

    if video_file is not None:
        # Display video info
        file_size_mb = len(video_file.getvalue()) / (1024 * 1024)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", video_file.name)
        with col2:
            st.metric("File Size", f"{file_size_mb:.1f} MB")
        with col3:
            st.metric("File Type", video_file.type)

        # Display video preview
        st.subheader("📺 Video Preview")
        st.video(video_file)
        
        # AI Analysis instructions
        st.info("""
        **🤖 AI Agents will analyze:**
        - 📱 Detection Agent: MediaPipe gesture detection
        - 🧠 Analysis Agent: AI-powered result interpretation
        - 💾 Data Agent: Intelligent storage and retrieval
        - 🎯 Coordinator Agent: Workflow orchestration
        
        **For best results:**
        - 💡 Ensure good lighting
        - 🤚 Keep medicine strip or bottle visible
        - 👄 Make clear hand-to-mouth movements
        - 🔄 Tilt your head slightly back when swallowing
        """)

        # AI Analysis button
        if st.button("🤖 Analyze with AI Agents", type="primary"):
            is_healthy, _ = check_api_health()
            if not is_healthy:
                st.error("❌ AI Agents API is not running. Please start the backend server first.")
                st.code("uvicorn ai_agents_api_backend:app --reload")
                return

            # File size check
            if file_size_mb > 50:  # 50MB limit
                st.error("File too large. Maximum size: 50MB")
                return

            # Process video with AI agents
            start_time = time.time()
            
            with st.spinner("🤖 AI Agents are analyzing your video... This involves multiple AI agents working together."):
                try:
                    # Prepare file for upload
                    files = {"file": (video_file.name, video_file.getvalue(), video_file.type)}
                    
                    # Make API request to AI agents
                    response = requests.post(ENDPOINTS["upload"], files=files, timeout=120)
                    
                    processing_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        st.success(f"✅ AI Agents analysis completed in {processing_time:.1f} seconds")
                        
                        # AI workflow status
                        if result.get("workflow_status") == "completed":
                            st.success("🤖 AI Agents workflow completed successfully")
                        
                        # Result summary with AI info
                        display_result_summary(result)
                        
                        # Detailed AI results
                        st.subheader("📊 Detailed AI Agents Analysis")
                        display_enhanced_results(result)
                        
                        # AI Explanation and recommendations
                        with st.expander("🧠 AI Agent Explanation & Recommendations", expanded=True):
                            explanation = result.get("explanation", "No explanation available.")
                            st.write(explanation)
                            
                            # Show which AI agent generated the explanation
                            if "analysis" in result and "analyzed_by" in result["analysis"]:
                                analyzed_by = result["analysis"]["analyzed_by"]
                                st.caption(f"🤖 Generated by: {analyzed_by.replace('_', ' ').title()}")
                        
                        # AI Reminder
                        with st.expander("🔔 AI Reminder & Next Steps"):
                            reminder = result.get("reminder", "No reminder available.")
                            st.write(reminder)
                        
                        # Download results option
                        if st.button("💾 Download AI Analysis Results"):
                            result_json = json.dumps(result, indent=2)
                            st.download_button(
                                label="📥 Download JSON",
                                data=result_json,
                                file_name=f"ai_agents_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )

                    else:
                        st.error(f"❌ AI Agents analysis failed (Status: {response.status_code})")
                        try:
                            error_detail = response.json().get("detail", "Unknown error")
                            st.error(f"Error details: {error_detail}")
                        except:
                            st.error(f"Server response: {response.text}")

                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. The AI agents might be busy or the video is too long.")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Connection error. Make sure the AI Agents API server is running.")
                except Exception as e:
                    st.error(f"💥 Unexpected error: {str(e)}")
                    logger.error(f"Streamlit error: {e}")

    # Test AI Agents with mock data
    st.markdown("---")
    st.subheader("🧪 Test AI Agents System")
    if st.button("🚀 Test AI Agents with Mock Data"):
        with st.spinner("Running AI agents system test..."):
            try:
                response = requests.post(ENDPOINTS["test"], timeout=10)
                if response.status_code == 200:
                    test_result = response.json()
                    st.success("✅ AI Agents system test successful!")
                    
                    if test_result.get("processed_by_ai_agents"):
                        st.success("🤖 All AI agents coordinated successfully")
                    
                    # Display test results
                    display_result_summary(test_result)
                    
                    with st.expander("AI Agents Test Results Details"):
                        st.json(test_result)
                else:
                    st.error(f"AI Agents test failed: {response.status_code}")
                    st.error(response.text)
            except Exception as e:
                st.error(f"AI Agents test error: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>🤖 AI Agents Medicine Intake Detection System v3.0 | Powered by Multi-Agent AI Architecture</p>
        <p>Features: Detection Agent • Analysis Agent • Scheduling Agent • Data Agent • Coordinator Agent</p>
        <p>Enhanced with: MediaPipe • OpenAI • Computer Vision • Intelligent Coordination</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
