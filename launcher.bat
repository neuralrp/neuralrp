@echo off
chcp 65001 >nul
title NeuralRP Launcher

echo.
echo ========================================
echo       NeuralRP - Roleplay Platform
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.8 or higher from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo.
    echo [2/3] Installing dependencies...
    echo This may take a few minutes on first run...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies!
        echo.
        pause
        exit /b 1
    )
) else (
    echo Dependencies already installed.
)

echo.
echo [3/3] Checking database...
python app/database_setup.py
if errorlevel 1 (
    echo.
    echo ERROR: Database setup failed! See error message above.
    echo.
    pause
    exit /b 1
)
echo Database ready.

echo.
echo Starting NeuralRP...
echo.
echo ========================================
echo  NOTE: First run downloads ~400MB AI model
echo  This takes 5-10 minutes. Please be patient.
echo ========================================
echo.
echo The application will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server.
echo.
echo ========================================
echo.

REM Start the application
python main.py

REM If server stops, keep window open to see any errors
if errorlevel 1 (
    echo.
    echo ========================================
    echo Server stopped with an error.
    echo ========================================
    pause
)
