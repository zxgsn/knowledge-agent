#!/bin/bash
# Knowledge Agent - One-click startup script
# Usage: bash start.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo "=========================================="
echo "  Knowledge Agent - Starting..."
echo "=========================================="

# 1. Start PostgreSQL via Docker Compose
echo ""
echo "[1/4] Starting PostgreSQL (Docker Compose)..."
cd "$BACKEND_DIR"
docker compose up -d postgres
echo "  Waiting for PostgreSQL to be ready..."
until docker compose exec -T postgres pg_isready -U knowledge_agent 2>/dev/null; do
  sleep 1
done
echo "  PostgreSQL is ready."

# 2. Start LangGraph backend
echo ""
echo "[2/4] Starting LangGraph backend on port 2024 (reload disabled for stable streaming)..."
cd "$BACKEND_DIR"
uv run langgraph dev --no-reload --port 2024 &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

# Wait for backend to be ready
echo "  Waiting for backend to be ready..."
until curl -s http://localhost:2024/ok >/dev/null 2>&1; do
  sleep 2
done
echo "  Backend is ready."

# 3. Start Library API server
echo ""
echo "[3/4] Starting Library API on port 8000..."
cd "$BACKEND_DIR"
uv run uvicorn src.api_server:app --port 8000 &
API_PID=$!
echo "  API PID: $API_PID"

# 4. Start frontend
echo ""
echo "[4/4] Starting frontend on port 5173..."
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!
echo "  Frontend PID: $FRONTEND_PID"

echo ""
echo "=========================================="
echo "  All services started!"
echo "  Frontend: http://localhost:5173/app/"
echo "  Backend:  http://localhost:2024"
echo "  Library:  http://localhost:5173/app/library"
echo "  API:      http://localhost:8000"
echo "  Press Ctrl+C to stop all services"
echo "=========================================="

# Trap Ctrl+C to stop all services
trap "echo ''; echo 'Stopping...'; kill $FRONTEND_PID $BACKEND_PID $API_PID 2>/dev/null; docker compose -f '$BACKEND_DIR/docker-compose.yml' down; exit 0" INT TERM

wait
