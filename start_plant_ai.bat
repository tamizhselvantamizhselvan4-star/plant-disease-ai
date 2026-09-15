@echo off
cd /d "C:\Users\ELCOT\Downloads\Plant_Disease_App"

echo Starting Plant Disease AI...
echo.

REM Check whether Plant Disease AI is already running
netstat -ano | findstr ":5000" | findstr "LISTENING" >nul

if %errorlevel%==0 (
    echo Plant Disease AI is already running.
    start "" "http://127.0.0.1:5000/?launch=home"
    exit /b
)

call ".venv\Scripts\activate.bat"

start "" "http://127.0.0.1:5000/?launch=home"

python app.py

pause