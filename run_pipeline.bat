@echo off
title Garmin Running Analytics Pipeline
echo ============================================================
echo   Ejecutando Garmin Running Analytics Pipeline...
echo ============================================================
echo.
cd /d "%~dp0"
.\.venv\Scripts\python.exe main.py
echo.
echo Presione cualquier tecla para cerrar...
pause >nul
