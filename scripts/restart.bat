@echo off
echo ========================================
echo MedBot Restart Script
echo ========================================
echo.

echo [1/3] Stopping old app...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING') do (
    echo Killing process %%a
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo [2/3] Starting MedBot...
echo.
echo ========================================
echo MedBot is starting...
echo Open: http://localhost:5000
echo Press Ctrl+C to stop
echo ========================================
echo.

cd ..
python app.py
