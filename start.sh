#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh  –  Build the React app then serve everything from FastAPI (port 8502)
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
echo -e "${GREEN}  AarogyaVaani – Building & Starting${RESET}"
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

# ── Build the React frontend ──────────────────────────────────────────────────
echo -e "\n${GREEN}▶ Building React frontend…${RESET}"
cd "$FRONTEND_DIR"
npm run build
echo -e "${GREEN}  ✓ Build complete → frontend/dist/${RESET}"

# ── Detect local network IP ───────────────────────────────────────────────────
NETWORK_IP=$(ipconfig getifaddr en0 2>/dev/null || \
             ipconfig getifaddr en1 2>/dev/null || \
             hostname -I 2>/dev/null | awk '{print $1}' || \
             echo "your-machine-ip")

# ── Trap to kill server on Ctrl+C ────────────────────────────────────────────
cleanup() {
    echo -e "\n${YELLOW}Shutting down…${RESET}"
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
    echo -e "${GREEN}Done.${RESET}"
}
trap cleanup INT TERM

# ── Start FastAPI (serves both API + built frontend) ─────────────────────────
echo -e "\n${GREEN}▶ Starting server on port 8502 (API + frontend)${RESET}"
cd "$BACKEND_DIR"
set -a; source .env 2>/dev/null; set +a
/opt/anaconda3/bin/uvicorn server:app \
    --host 0.0.0.0 \
    --port 8502 &
API_PID=$!

sleep 2   # give uvicorn a moment to bind

echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "  Local  : ${GREEN}http://localhost:8502/${RESET}"
echo -e "  Network: ${GREEN}http://${NETWORK_IP}:8502/${RESET}  ← share this"
echo -e "  API docs: http://localhost:8502/docs"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "  Press ${YELLOW}Ctrl+C${RESET} to stop.\n"

# ── Wait ──────────────────────────────────────────────────────────────────────
wait "$API_PID"
