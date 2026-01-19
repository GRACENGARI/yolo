@echo off
echo Starting HAWKEYE CV Intelligence Engine...
echo Ensure you have installed requirements (pip install -r cv_engine\requirements.txt)
echo.
..\backend\venv\Scripts\python.exe -m cv_engine.stream_processor
pause
