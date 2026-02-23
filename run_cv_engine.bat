@echo off
echo Starting HAWKEYE CV Intelligence Engine (Dockerized) // Phase 4...
echo.
echo Usage Examples:
echo   - Run with Docker: run_cv_engine.bat --docker
echo   - Run Locally (requires Python 3.10): run_cv_engine.bat --local
echo.

set PYTHON_EXE=..\backend\venv\Scripts\python.exe

if "%1"=="--docker" (
    echo Building and running CV Engine in Docker...
    docker-compose -f ..\backend\docker-compose.yml up --build -d cv-engine
    echo Container started. Logs:
    docker logs -f hawkeye_cv_engine
) else (
    echo [DEFAULT] Running on local Python environment (Requires Python 3.10)
    %PYTHON_EXE% -m cv_engine.stream_processor %*
)

pause
