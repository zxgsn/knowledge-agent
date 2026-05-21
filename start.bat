@echo off
REM Knowledge Agent - One-click startup script (Windows)
REM Usage: start.bat

setlocal

set SCRIPT_DIR=%~dp0
set BACKEND_DIR=%SCRIPT_DIR%backend
set FRONTEND_DIR=%SCRIPT_DIR%frontend

echo ==========================================
echo   Knowledge Agent - Starting...
echo ==========================================

REM Check prerequisites
where uv >nul 2>&1
if errorlevel 1 (
  echo ERROR: uv is not installed or not in PATH
  echo Install from: https://docs.astral.sh/uv/
  pause
  exit /b 1
)

where docker >nul 2>&1
if errorlevel 1 (
  echo ERROR: docker is not installed or not in PATH
  pause
  exit /b 1
)

REM Check if Docker daemon is running
docker info >nul 2>&1
if errorlevel 1 (
  echo Docker Desktop is not running. Attempting to start...
  if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" (
    start "" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
  ) else if exist "%LOCALAPPDATA%\Programs\Docker\Docker\Docker Desktop.exe" (
    start "" "%LOCALAPPDATA%\Programs\Docker\Docker\Docker Desktop.exe"
  ) else (
    echo ERROR: Cannot find Docker Desktop. Please start it manually.
    pause
    exit /b 1
  )
  echo   Waiting for Docker Desktop to start...
  :wait_docker
  timeout /t 5 /nobreak >nul
  docker info >nul 2>&1
  if errorlevel 1 goto wait_docker
  echo   Docker Desktop is ready.
)

REM 1. Start PostgreSQL via Docker Compose
echo.
echo [1/3] Starting PostgreSQL (Docker Compose)...
cd /d "%BACKEND_DIR%"
docker compose up -d postgres
if errorlevel 1 (
  echo ERROR: Failed to start PostgreSQL
  pause
  exit /b 1
)
echo   Waiting for PostgreSQL to be ready...
timeout /t 5 /nobreak >nul
:wait_pg
docker compose exec -T postgres pg_isready -U knowledge_agent >nul 2>&1
if errorlevel 1 (
  timeout /t 1 /nobreak >nul
  goto wait_pg
)
echo   PostgreSQL is ready.

REM 2. Start LangGraph backend
echo.
echo [2/3] Starting LangGraph backend on port 2024...
cd /d "%BACKEND_DIR%"
start "Knowledge Agent Backend" cmd /k "uv run langgraph dev --port 2024"
echo   Waiting for backend to be ready (max 120 seconds)...
set BACKEND_TIMEOUT=0
:wait_backend
timeout /t 3 /nobreak >nul
set /a BACKEND_TIMEOUT+=3
if %BACKEND_TIMEOUT% geq 120 (
  echo ERROR: Backend failed to start within 120 seconds
  echo Please check the backend window for errors.
  pause
  exit /b 1
)
curl -s http://localhost:2024/ok >nul 2>&1
if errorlevel 1 goto wait_backend
echo   Backend is ready.

REM 3. Start frontend
echo.
echo [3/3] Starting frontend on port 5173...
cd /d "%FRONTEND_DIR%"
start "Knowledge Agent Frontend" cmd /k "npm run dev"

echo.
echo ==========================================
echo   All services started!
echo   Frontend: http://localhost:5173/app/
echo   Backend:  http://localhost:2024
echo.
echo   Close the terminal windows to stop.
echo ==========================================

pause
