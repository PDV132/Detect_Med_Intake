@echo off
echo 🚀 Setting up Medicine Intake Detection System...

:: Create virtual environment
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate

:: Upgrade pip
pip install --upgrade pip

:: Install requirements
pip install -r requirements.txt

:: Create necessary directories
if not exist "uploads" mkdir uploads
if not exist "temp" mkdir temp
if not exist "logs" mkdir logs
if not exist "output_clips" mkdir "output_clips"

:: Create .env file if it doesn't exist
if not exist ".env" (
    echo Creating .env file...
    echo OPENAI_API_KEY=your_openai_api_key_here > .env
    echo FASTAPI_HOST=localhost >> .env
    echo FASTAPI_PORT=8000 >> .env
    echo LOG_LEVEL=INFO >> .env
    echo ⚠️  Please edit .env file and add your OpenAI API key
)

echo ✅ Setup complete!
echo.
echo Next steps:
echo 1. Edit .env file and add your OpenAI API key
echo 2. Start the backend: uvicorn api_backend:app --reload
echo 3. Start the frontend: streamlit run app.py

pause