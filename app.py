import streamlit as st
import requests
import time
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_URL = "http://localhost:8000/upload-video/"
TEST_API_URL = "http://localhost:8000/test-detection/"
HEALTH_CHECK_URL = "http://localhost:8000/health"

# Page configuration
st.set_page_config(
    page_title="Medicine Intake Detection",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_api_health():
    """Check if the FastAPI backend is running"""
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def display_result_summary(result):
    """Display a summary of the detection results"""
    detection = result.get("detection", {})
    analysis = result.get("analysis", {})
    
    status = detection.get("status", "unknown")
    
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

def display_detailed_results(result):
    """Display detailed results in expandable sections"""
    
    # Detection Results
    with st.expander("🔍 Detection Results", expanded=True):
        detection = result.get("detection", {})
        st.json(detection)
        
        # Additional detection info
        if detection.get("status") == "taken":
            events = detection.get("events", [])
            if events:
                st.write("**Detected Events:**")
                for i, event in enumerate(events):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Timestamp", f"{event.get('timestamp_sec', 0):.1f}s")
                    with col2:
                        st.metric("Head Sideways", "Yes" if event.get('head_sideways') else "No")
                    with col3:
                        st.metric("Head Backward", "Yes" if event.get('head_bent_backward') else "No")

    # Analysis Results
    with st.expander("🤖 AI Analysis", expanded=True):
        analysis = result.get("analysis", {})
        
        status = analysis.get("status", "unknown")
        confidence = analysis.get("confidence", "unknown")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Analysis Status", status.title())
        with col2:
            st.metric("Confidence Level", confidence.title())
        
        message = analysis.get("message", "No analysis message available.")
        st.write("**Analysis Message:**")
        st.write(message)

    # Explanation
    with st.expander("💡 Explanation & Tips"):
        explanation = result.get("explanation", "No explanation available.")
        st.write(explanation)

    # Reminder
    with st.expander("🔔 Reminder"):
        reminder = result.get("reminder", "No reminder available.")
        st.write(reminder)

def main():
    st.title("💊 Medicine Intake Detection & Explanation")
    st.markdown("Upload a video to detect and analyze medicine intake gestures using AI.")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # API Health Check
        st.subheader("System Status")
        if st.button("Check API Status"):
            with st.spinner("Checking API..."):
                if check_api_health():
                    st.success("✅ API is running")
                else:
                    st.error("❌ API is not accessible")
                    st.error("Make sure to run: `uvicorn api_backend:app --reload`")
        
        # Test API
        st.subheader("Testing")
        if st.button("Test with Mock Data"):
            with st.spinner("Running test..."):
                try:
                    response = requests.post(TEST_API_URL, timeout=10)
                    if response.status_code == 200:
                        test_result = response.json()
                        st.success("✅ Test successful")
                        with st.expander("Test Results"):
                            st.json(test_result)
                    else:
                        st.error(f"Test failed: {response.status_code}")
                        st.error(response.text)
                except Exception as e:
                    st.error(f"Test error: {str(e)}")
        
        # Settings
        st.subheader("Upload Settings")
        max_file_size = st.slider("Max File Size (MB)", 1, 100, 25)
        st.info(f"Maximum file size: {max_file_size}MB")

    # Main content
    # Input method selection
    input_method = st.radio(
        "Select input method:",
        ("Upload Video File", "Use Webcam (Coming Soon)"),
        help="Choose how you want to provide the video for analysis"
    )

    if input_method == "Upload Video File":
        st.subheader("📁 Upload Video")
        
        # File uploader
        video_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP4, AVI, MOV, MKV (Max 2 minutes)"
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

            # Analysis button
            if st.button("🔍 Analyze Video", type="primary"):
                if not check_api_health():
                    st.error("❌ API is not running. Please start the backend server first.")
                    st.code("uvicorn api_backend:app --reload")
                    return

                # File size check
                if file_size_mb > max_file_size:
                    st.error(f"File too large. Maximum size: {max_file_size}MB")
                    return

                # Process video
                start_time = time.time()
                
                with st.spinner("🔄 Uploading and analyzing video... This may take a few moments."):
                    try:
                        # Prepare file for upload
                        files = {"file": (video_file.name, video_file.getvalue(), video_file.type)}
                        
                        # Make API request
                        response = requests.post(API_URL, files=files, timeout=60)
                        
                        processing_time = time.time() - start_time
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Display results
                            st.success(f"✅ Analysis completed in {processing_time:.1f} seconds")
                            
                            # Result summary
                            display_result_summary(result)
                            
                            # Detailed results
                            st.subheader("📊 Detailed Results")
                            display_detailed_results(result)
                            
                            # Download results option
                            if st.button("💾 Download Results as JSON"):
                                result_json = json.dumps(result, indent=2)
                                st.download_button(
                                    label="Download JSON",
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

    else:  # Webcam option
        st.subheader("📷 Webcam Recording")
        st.info("🚧 Webcam functionality is coming soon! For now, please use video file upload.")
        
        # Placeholder for future webcam integration
        with st.expander("Future Features"):
            st.write("- Live webcam recording")
            st.write("- Real-time gesture detection")
            st.write("- Automatic recording triggers")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>Medicine Intake Detection System | Powered by MediaPipe & AI</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()