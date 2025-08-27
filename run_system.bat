@echo off
echo 🚀 Starting Medicine Intake Detection System...

:: Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate

:: Start backend
echo 🔧 Starting FastAPI backend...
start "Backend" cmd /c "uvicorn api_backend:app --reload --host 0.0.0.0 --port 8000"

:: Wait for backend to start
timeout /t 5 /nobreak > nul

:: Start frontend
echo 🎨 Starting Streamlit frontend...
streamlit run app.py

pause