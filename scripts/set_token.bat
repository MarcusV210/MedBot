@echo off
echo ========================================
echo MedBot - Set GitHub Token
echo ========================================
echo.
echo This will set your GitHub token for this session.
echo Get your token from: https://github.com/settings/tokens
echo.
set /p TOKEN="Enter your GitHub token: "
echo.
echo Setting GITHUB_TOKEN environment variable...
set GITHUB_TOKEN=%TOKEN%
echo.
echo ✓ Token set for this session!
echo.
echo Now run: start.bat
echo.
pause
