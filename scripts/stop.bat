@echo off
echo ========================================
echo Stopping MedBot
echo ========================================
echo.

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING') do (
    echo Stopping process %%a...
    taskkill /F /PID %%a
)

echo.
echo MedBot stopped!
echo.
pause
