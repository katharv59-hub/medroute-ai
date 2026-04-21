@echo off
title MedRoute AI — Production Detection System v3.0
echo ============================================
echo   MedRoute AI v3.0 — Starting System
echo ============================================
echo.
echo   Performance Mode: %PERF_MODE%
echo   (Set PERF_MODE=LOW, BALANCED, or HIGH in .env)
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
