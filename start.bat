@echo off
echo Starting Agentic RAG Assistant...
cd /d "%~dp0"

if not exist ".env" (
    echo ERROR: .env file not found. Copy .env.example to .env and fill in your API keys.
    pause
    exit /b 1
)

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt --quiet

echo Starting server...
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

pause
