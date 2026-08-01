@echo off
title WhatsApp Chatbot FastAPI Server
echo ===================================================
echo 🚀 Starting WhatsApp Chatbot Local FastAPI Server...
echo ===================================================
echo.

:: Check if virtual environment exists
if not exist "%~dp0.venv" (
    echo ❌ ERROR: Virtual environment '.venv' not found in this folder!
    echo Please create it first.
    pause
    exit /b
)

:: Activate virtual environment
echo [1/2] Activating Python Virtual Environment...
call "%~dp0.venv\Scripts\activate.bat"

:: Run the FastAPI server
echo [2/2] Launching Webhook Server via Uvicorn...
echo.
echo Server is launching on: http://localhost:5533
echo Webhook endpoints:
echo   - GET/POST http://localhost:5533/webhook
echo.
echo Press Ctrl+C to terminate the server.
echo ===================================================
echo.

uvicorn main:app --host 0.0.0.0 --port 5533 --reload

pause
