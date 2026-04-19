@echo off
SET URL=http://127.0.0.1:8000
TITLE Magika AI Server

echo [1/2] Starting Backend...
:: Running through python module to ensure correct environment
start "MagikaServer" cmd /k "python -m uvicorn main:app --host 127.0.0.1 --port 8000"

echo [2/2] Opening Chrome...
timeout /t 5 /nobreak >nul
start chrome %URL%

echo Server is active. Keep the console window open.
pause