# enhanced_app.py
import streamlit as st
import requests
import time
import json
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
    "start_monitoring": f"{API_URL}/start-live-monitoring/",
    "stop_monitoring": f"{API_URL}/stop-monitoring/",
    "monitoring_status": f"{API_URL}/monitoring-status/",
    "schedule": f"{API_URL}/schedule-monitoring/",
    "scheduled_times": f"{API_URL}/scheduled-times/",
    "clear_schedule": f"{API_URL}/clear-schedule/",
    "monitoring_results": f"{API_URL}/monitoring-results/"
}

# Page configuration
st.set_page_config(
    page_title="Enhanced Medicine Intake Detection",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_api_health():
    """Check if the FastAPI backend is running"""
    try:
        response = requests.get(ENDPOINTS["health"], timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return False, None

def get_monitoring_status():
    """Get current monitoring status"""
    try:
        response = requests.get(ENDPOINTS["monitoring_status"], timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Failed to get monitoring status: {e}")
    return None

def start_live_monitoring(duration: int):
    """Start live monitoring"""
    try:
        payload = {"duration": duration}
        response = requests.post(ENDPOINTS["start_monitoring"], json=payload, timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def stop_monitoring():
    """Stop active monitoring"""
    try:
        response = requests.post(ENDPOINTS["stop_monitoring"], timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def schedule_monitoring(time_str: str, duration: int):
    """Schedule daily monitoring"""
    try:
        payload = {"time": time_str, "duration": duration}
        response = requests.post(ENDPOINTS["schedule"], json=payload, timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def get_monitoring_results():
    """Get monitoring results"""
    try:
        response = requests.get(ENDPOINTS["monitoring_results"], timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Failed to get monitoring results: {e}")
    return None

def display_result_summary(result):
    """Display a summary of the detection results"""
    detection = result.get("detection", {})
    analysis = result.get("analysis", {})
    
    status = detection.get("status", "unknown")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if status == "taken":
            st.success("🎉 Medicine Intake Detected!")
            events = detection.get("events", [])
            if events:
                st.info(f"Detected at {events[0].get('timestamp_sec', 0):.1f} seconds")
        elif status == "missed":
            st.warning("⚠️ Medicine Intake Not Detected")
        elif status == "error":
            st.error("❌ Detection Error")
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
        # Confidence score
        confidence = analysis.get("confidence", "unknown")
        if confidence == "high":
            st.success(f"📊 Confidence: {confidence.title()}")
        elif confidence == "medium":
            st.warning(f"📊 Confidence: {confidence.title()}")
        else:
            st.error(f"📊 Confidence: {confidence.title()}")

def display_enhanced_results(result):
    """Display enhanced results with medicine object detection info"""
    
    # Detection Results
    with st.expander("🔍 Enhanced Detection Results", expanded=True):
        detection = result.get("detection", {})
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", detection.get("status", "unknown").title())
        with col2:
            st.metric("Processing Time", f"{detection.get('processing_time', 0):.1f}s")
        with col3:
            st.metric("File Size", f"{detection.get('file_size_mb', 0):.1f}MB")
        
        # Event details
        events = detection.get("events", [])
        if events:
            st.write("**Detection Events:**")
            for i, event in enumerate(events):
                st.write(f"**Event {i+1}:**")
                
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
                
                # Confidence scores
                confidence_scores = details.get("confidence_scores", {})
                if confidence_scores:
                    st.write("**Confidence Scores:**")
                    score_cols = st.columns(len(confidence_scores))
                    for idx, (key, score) in enumerate(confidence_scores.items()):
                        with score_cols[idx]:
                            st.metric(key.replace("_", " ").title(), f"{score:.0%}")

def display_monitoring_dashboard():
    """Display live monitoring dashboard"""
    st.header("📹 Live Monitoring Dashboard")
    
    # Get current status
    status = get_monitoring_status()
    
    if status:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if status.get("monitoring_active"):
                st.success("🔴 Live Monitoring Active")
            else:
                st.info("⭕ Monitoring Inactive")
        
        with col2:
            if status.get("scheduler_active"):
                st.success("⏰ Scheduler Running")
            else:
                st.warning("⏰ Scheduler Stopped")
        
        with col3:
            scheduled_count = len(status.get("scheduled_times", []))
            st.metric("Scheduled Times", scheduled_count)
    
    # Live monitoring controls
    st.subheader("🎬 Start Live Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.slider("Monitoring Duration (seconds)", 30, 300, 60)
        
        if st.button("▶️ Start Live Monitoring", type="primary"):
            with st.spinner("Starting live monitoring..."):
                success, response = start_live_monitoring(duration)
                
                if success:
                    st.success(f"✅ Live monitoring started for {duration} seconds!")
                    st.json(response)
                else:
                    st.error(f"❌ Failed to start monitoring: {response.get('error', 'Unknown error')}")
    
    with col2:
        if st.button("⏹️ Stop Monitoring", type="secondary"):
            success, response = stop_monitoring()
            
            if success:
                st.success("✅ Monitoring stopped!")
            else:
                st.error(f"❌ Failed to stop monitoring: {response.get('error', 'Unknown error')}")

def display_scheduling_interface():
    """Display scheduling interface"""
    st.header("⏰ Schedule Daily Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Add New Schedule")
        
        # Time input
        schedule_time = st.time_input(
            "Monitoring Time",
            value=datetime.strptime("09:00", "%H:%M").time(),
            help="Select the time for daily monitoring"
        )
        
        schedule_duration = st.slider(
            "Duration (seconds)", 
            30, 300, 60,
            help="How long to monitor each day"
        )
        
        if st.button("➕ Add Schedule", type="primary"):
            time_str = schedule_time.strftime("%H:%M")
            success, response = schedule_monitoring(time_str, schedule_duration)
            
            if success:
                st.success(f"✅ Monitoring scheduled for {time_str} daily!")
                st.rerun()
            else:
                st.error(f"❌ Failed to schedule: {response.get('error', 'Unknown error')}")
    
    with col2:
        st.subheader("Current Schedule")
        
        # Get scheduled times
        try:
            response = requests.get(ENDPOINTS["scheduled_times"], timeout=5)
            if response.status_code == 200:
                schedule_data = response.json()
                scheduled_times = schedule_data.get("scheduled_times", [])
                
                if scheduled_times:
                    for schedule in scheduled_times:
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"⏰ {schedule['time']} - {schedule['duration']}s")
                        # Note: Individual removal would need additional API endpoint
                    
                    if st.button("🗑️ Clear All Schedules", type="secondary"):
                        try:
                            clear_response = requests.delete(ENDPOINTS["clear_schedule"], timeout=5)
                            if clear_response.status_code == 200:
                                st.success("✅ All schedules cleared!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to clear schedules")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                else:
                    st.info("No scheduled monitoring times")
            else:
                st.error("Failed to fetch schedule")
        except Exception as e:
            st.error(f"Error fetching schedule: {e}")

def display_monitoring_results():
    """Display monitoring results and analytics"""
    st.header("📊 Monitoring Results & Analytics")
    
    results_data = get_monitoring_results()
    
    if not results_data or not results_data.get("results"):
        st.info("No monitoring results available yet")
        return
    
    results = results_data["results"]
    
    # Summary metrics
    st.subheader("📈 Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_sessions = len(results)
    successful_detections = sum(1 for r in results if r.get("medicine_intake_detected", False))
    success_rate = (successful_detections / total_sessions * 100) if total_sessions > 0 else 0
    
    with col1:
        st.metric("Total Sessions", total_sessions)
    with col2:
        st.metric("Successful Detections", successful_detections)
    with col3:
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col4:
        st.metric("Recent Results", len([r for r in results if datetime.fromisoformat(r["monitoring_date"].replace('Z', '+00:00')).date() == datetime.now().date()]))
    
    # Results timeline
    st.subheader("📅 Detection Timeline")
    
    if results:
        # Prepare data for visualization
        timeline_data = []
        for result in results:
            timeline_data.append({
                "Date": datetime.fromisoformat(result["monitoring_date"].replace('Z', '+00:00')).date(),
                "Time": datetime.fromisoformat(result["monitoring_date"].replace('Z', '+00:00')).time(),
                "Detected": result.get("medicine_intake_detected", False),
                "Duration": result.get("duration_seconds", 0),
                "Events": result.get("total_detections", 0)
            })
        
        df = pd.DataFrame(timeline_data)
        
        # Create timeline chart
        fig = px.scatter(df, 
                        x="Date", 
                        y="Time",
                        color="Detected",
                        size="Events",
                        hover_data=["Duration"],
                        title="Medicine Intake Detection Timeline")
        
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
                      title="Daily Success Rate (%)",
                      markers=True)
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Detailed results
    st.subheader("📋 Detailed Results")
    
    for i, result in enumerate(results[:10]):  # Show last 10 results
        with st.expander(f"Session {i+1}: {result['monitoring_date'][:19]}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Detection Status", 
                         "✅ Detected" if result.get("medicine_intake_detected") else "❌ Not Detected")
            with col2:
                st.metric("Duration", f"{result.get('duration_seconds', 0)}s")
            with col3:
                st.metric("Events", result.get("total_detections", 0))
            
            if result.get("detection_events"):
                st.write("**Detection Events:**")
                for event in result["detection_events"]:
                    st.json(event)

def main():
    st.title("💊 Enhanced Medicine Intake Detection System")
    st.markdown("Advanced AI-powered medicine intake detection with live monitoring, scheduling, and object recognition.")
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ System Controls")
        
        # API Health Check
        st.subheader("System Status")
        if st.button("🔄 Check System Status"):
            with st.spinner("Checking system..."):
                is_healthy, health_data = check_api_health()
                if is_healthy:
                    st.success("✅ System is running")
                    if health_data:
                        st.json(health_data)
                else:
                    st.error("❌ System is not accessible")
                    st.error("Make sure to run: `uvicorn enhanced_api_backend:app --reload`")
        
        # Navigation
        st.subheader("📍 Navigation")
        page = st.selectbox(
            "Select Page",
            ["🎬 Video Upload", "📹 Live Monitoring", "⏰ Scheduling", "📊 Results & Analytics"],
            help="Choose which feature to use"
        )
    
    # Main content based on selected page
    if page == "🎬 Video Upload":
        display_video_upload_page()
    elif page == "📹 Live Monitoring":
        display_monitoring_dashboard()
    elif page == "⏰ Scheduling":
        display_scheduling_interface()
    elif page == "📊 Results & Analytics":
        display_monitoring_results()

def display_video_upload_page():
    """Display the video upload page"""
    st.header("🎬 Video Upload & Analysis")
    
    # File uploader
    video_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a video showing medicine intake. Make sure the medicine strip/bottle is visible!"
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
        
        # Analysis instructions
        st.info("""
        **For best results:**
        - 📱 Hold your phone/camera steady
        - 💡 Ensure good lighting
        - 🤚 Keep medicine strip or bottle visible in one hand
        - 👄 Make clear hand-to-mouth movements
        - 🔄 Tilt your head slightly back when swallowing
        """)

        # Analysis button
        if st.button("🔍 Analyze Video", type="primary"):
            is_healthy, _ = check_api_health()
            if not is_healthy:
                st.error("❌ API is not running. Please start the backend server first.")
                st.code("uvicorn enhanced_api_backend:app --reload")
                return

            # File size check
            if file_size_mb > 50:  # 50MB limit
                st.error("File too large. Maximum size: 50MB")
                return

            # Process video
            start_time = time.time()
            
            with st.spinner("🔄 Analyzing video with enhanced detection... This may take a few moments."):
                try:
                    # Prepare file for upload
                    files = {"file": (video_file.name, video_file.getvalue(), video_file.type)}
                    
                    # Make API request
                    response = requests.post(ENDPOINTS["upload"], files=files, timeout=120)
                    
                    processing_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        st.success(f"✅ Analysis completed in {processing_time:.1f} seconds")
                        
                        # Result summary
                        display_result_summary(result)
                        
                        # Detailed results
                        st.subheader("📊 Detailed Analysis Results")
                        display_enhanced_results(result)
                        
                        # Explanation and recommendations
                        with st.expander("💡 AI Explanation & Recommendations", expanded=True):
                            explanation = result.get("explanation", "No explanation available.")
                            st.write(explanation)
                        
                        # Reminder
                        with st.expander("🔔 Reminder & Next Steps"):
                            reminder = result.get("reminder", "No reminder available.")
                            st.write(reminder)
                        
                        # Download results option
                        if st.button("💾 Download Results as JSON"):
                            result_json = json.dumps(result, indent=2)
                            st.download_button(
                                label="📥 Download JSON",
                                data=result_json,
                                file_name=f"medicine_detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )

                    else:
                        st.error(f"❌ Analysis failed (Status: {response.status_code})")
                        try:
                            error_detail = response.json().get("detail", "Unknown error")
                            st.error(f"Error details: {error_detail}")
                        except:
                            st.error(f"Server response: {response.text}")

                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. The video might be too long or the server is busy.")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Connection error. Make sure the API server is running.")
                except Exception as e:
                    st.error(f"💥 Unexpected error: {str(e)}")
                    logger.error(f"Streamlit error: {e}")

    # Test with mock data
    st.markdown("---")
    st.subheader("🧪 Test System")
    if st.button("🚀 Test with Mock Data"):
        with st.spinner("Running system test..."):
            try:
                response = requests.post(ENDPOINTS["test"], timeout=10)
                if response.status_code == 200:
                    test_result = response.json()
                    st.success("✅ System test successful!")
                    
                    # Display test results
                    display_result_summary(test_result)
                    
                    with st.expander("Test Results Details"):
                        st.json(test_result)
                else:
                    st.error(f"Test failed: {response.status_code}")
                    st.error(response.text)
            except Exception as e:
                st.error(f"Test error: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>Enhanced Medicine Intake Detection System v2.0 | Powered by MediaPipe, AI & Computer Vision</p>
        <p>Features: Live Monitoring • Daily Scheduling • Object Detection • Gesture Recognition</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()