@echo off
title ASEP — Ambulance Signal Emergency Priority
echo ============================================
echo   ASEP — Starting Detection System
echo ============================================
echo.

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist "..\..\..\ev_project\venv311\Scripts\activate.bat" (
    call ..\..\..\ev_project\venv311\Scripts\activate.bat
) else (
    echo [WARN] No venv found, using system Python
)

echo Starting detection...
python detect.py
pause
