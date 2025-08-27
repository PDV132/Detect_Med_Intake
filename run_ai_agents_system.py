"""
AI Agents System Startup Script
Comprehensive startup script for the AI Agents Medicine Intake Detection System
"""

import os
import sys
import subprocess
import time
import logging
import asyncio
import threading
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"Python version: {sys.version}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'streamlit', 'requests', 'numpy', 
        'mediapipe', 'opencv-python', 'langchain', 'schedule'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"❌ {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install missing packages with: pip install -r ai_agents_requirements.txt")
        return False
    
    return True

def check_environment():
    """Check environment setup"""
    logger.info("Checking environment setup...")
    
    # Check for OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        logger.info("✅ OpenAI API key found")
    else:
        logger.warning("⚠️ OpenAI API key not found. AI explanations will use fallback responses.")
        logger.info("Set OPENAI_API_KEY environment variable for enhanced AI features")
    
    # Check directories
    directories = ["uploads", "monitoring_results", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✅ Directory '{directory}' ready")
    
    return True

def test_ai_agents_system():
    """Test the AI agents system"""
    logger.info("Testing AI Agents System...")
    
    try:
        # Import and test the AI agents system
        from ai_agents_system import AIAgentsSystem
        
        async def test_system():
            system = AIAgentsSystem()
            try:
                await system.start()
                logger.info("✅ AI Agents System started successfully")
                
                # Test system status
                status = await system.get_system_status()
                logger.info(f"✅ System status: {len(status)} agents running")
                
                await system.stop()
                logger.info("✅ AI Agents System stopped successfully")
                return True
                
            except Exception as e:
                logger.error(f"❌ AI Agents System test failed: {e}")
                return False
        
        # Run the async test
        result = asyncio.run(test_system())
        return result
        
    except ImportError as e:
        logger.error(f"❌ Failed to import AI Agents System: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ AI Agents System test failed: {e}")
        return False

def start_api_server():
    """Start the AI Agents API server"""
    logger.info("Starting AI Agents API Server...")
    
    try:
        # Start the FastAPI server
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "ai_agents_api_backend:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        import requests
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ AI Agents API Server is running")
                logger.info("🌐 API available at: http://localhost:8000")
                logger.info("📚 API docs at: http://localhost:8000/docs")
                return process
            else:
                logger.error(f"❌ API Server health check failed: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Cannot connect to API Server: {e}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to start API Server: {e}")
        return None

def start_streamlit_app():
    """Start the AI Agents Streamlit app"""
    logger.info("Starting AI Agents Streamlit App...")
    
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "ai_agents_app.py", 
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        
        # Wait a moment for app to start
        time.sleep(5)
        
        logger.info("✅ AI Agents Streamlit App is starting")
        logger.info("🎨 Web interface available at: http://localhost:8501")
        return process
        
    except Exception as e:
        logger.error(f"❌ Failed to start Streamlit App: {e}")
        return None

def display_system_info():
    """Display system information"""
    print("\n" + "="*60)
    print("🤖 AI AGENTS MEDICINE INTAKE DETECTION SYSTEM")
    print("="*60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Working directory: {os.getcwd()}")
    print("\n🤖 AI AGENTS:")
    print("  • Detection Agent - MediaPipe gesture detection")
    print("  • Analysis Agent - AI-powered result analysis")
    print("  • Scheduling Agent - Smart scheduling management")
    print("  • Data Agent - Intelligent data storage and analytics")
    print("  • Coordinator Agent - Workflow orchestration")
    print("\n🌐 SERVICES:")
    print("  • API Server: http://localhost:8000")
    print("  • API Docs: http://localhost:8000/docs")
    print("  • Web Interface: http://localhost:8501")
    print("="*60)

def main():
    """Main startup function"""
    print("🚀 Starting AI Agents Medicine Intake Detection System...")
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    if not check_dependencies():
        print("\n💡 To install dependencies, run:")
        print("pip install -r ai_agents_requirements.txt")
        sys.exit(1)
    
    if not check_environment():
        sys.exit(1)
    
    # Test AI agents system
    if not test_ai_agents_system():
        logger.error("❌ AI Agents System test failed. Please check the installation.")
        sys.exit(1)
    
    # Display system information
    display_system_info()
    
    # Start services
    api_process = None
    streamlit_process = None
    
    try:
        # Start API server
        api_process = start_api_server()
        if not api_process:
            logger.error("❌ Failed to start API server")
            sys.exit(1)
        
        # Start Streamlit app
        streamlit_process = start_streamlit_app()
        if not streamlit_process:
            logger.error("❌ Failed to start Streamlit app")
            if api_process:
                api_process.terminate()
            sys.exit(1)
        
        print("\n✅ AI Agents System is running!")
        print("\n📖 USAGE:")
        print("  1. Open http://localhost:8501 for the web interface")
        print("  2. Upload videos for AI analysis")
        print("  3. Set up live monitoring schedules")
        print("  4. View analytics and results")
        print("\n⚠️  Press Ctrl+C to stop all services")
        
        # Keep the script running
        try:
            while True:
                time.sleep(1)
                
                # Check if processes are still running
                if api_process.poll() is not None:
                    logger.error("❌ API server stopped unexpectedly")
                    break
                    
                if streamlit_process.poll() is not None:
                    logger.error("❌ Streamlit app stopped unexpectedly")
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Shutting down AI Agents System...")
            
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        
    finally:
        # Clean up processes
        if api_process:
            logger.info("Stopping API server...")
            api_process.terminate()
            api_process.wait()
            
        if streamlit_process:
            logger.info("Stopping Streamlit app...")
            streamlit_process.terminate()
            streamlit_process.wait()
            
        print("✅ AI Agents System stopped successfully")

if __name__ == "__main__":
    main()
