#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh  –  Launch both the React landing page and Streamlit backend
# Usage: bash start.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
BACKEND_DIR="$SCRIPT_DIR"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}  Indic Mental Health Screening – Starting services${RESET}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

# ── Check for .env ────────────────────────────────────────────────────────────
if [ ! -f "$BACKEND_DIR/.env" ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Create one with OPENAI_API_KEY=sk-...${RESET}"
    exit 1
fi

# ── Check for node_modules ────────────────────────────────────────────────────
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo -e "${YELLOW}📦 Installing frontend dependencies...${RESET}"
    cd "$FRONTEND_DIR" && npm install
fi

# ── Trap to kill both servers on Ctrl+C ──────────────────────────────────────
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${RESET}"
    kill "$API_PID" "$FRONTEND_PID" 2>/dev/null || true
    wait "$API_PID" "$FRONTEND_PID" 2>/dev/null || true
    echo -e "${GREEN}Done.${RESET}"
}
trap cleanup INT TERM

# ── Start FastAPI (chat API) ───────────────────────────────────────────────────
echo -e "\n${GREEN}▶ Starting FastAPI chat API on http://localhost:8502${RESET}"
cd "$BACKEND_DIR"
# Export env vars from .env
set -a; source .env 2>/dev/null; set +a
/opt/anaconda3/bin/uvicorn server:app \
    --host 0.0.0.0 \
    --port 8502 \
    > /tmp/api.log 2>&1 &
API_PID=$!

# ── Start React (frontend) ────────────────────────────────────────────────────
echo -e "${GREEN}▶ Starting React frontend on http://localhost:3000${RESET}"
cd "$FRONTEND_DIR"
npm run dev -- --port 3000 &
FRONTEND_PID=$!

echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "  Landing page : ${GREEN}http://localhost:3000${RESET}"
echo -e "  Chat API     : ${GREEN}http://localhost:8502/docs${RESET}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "  Press ${YELLOW}Ctrl+C${RESET} to stop all servers.\n"

# ── Wait ──────────────────────────────────────────────────────────────────────
wait "$API_PID" "$FRONTEND_PID"
