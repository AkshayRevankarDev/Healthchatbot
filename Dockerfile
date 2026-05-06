# ─────────────────────────────────────────────────────────────
#  AarogyaVaani — Hugging Face Spaces  (Docker SDK, port 7860)
# ─────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── System deps: Node 18, ffmpeg, git, build tools ───────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ffmpeg build-essential ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (cached layer) ───────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── IndicTransToolkit (separate repo by VarunGumma) ──────────
RUN pip install --no-cache-dir \
    "git+https://github.com/VarunGumma/IndicTransToolkit.git"

# ── Frontend: install deps (cached layer) ────────────────────
COPY frontend/package*.json frontend/
RUN cd frontend && npm ci

# ── Copy all source ───────────────────────────────────────────
COPY . .

# ── Build React frontend → frontend/dist/ ─────────────────────
RUN cd frontend && npm run build

# HF Spaces listens on 7860
EXPOSE 7860

# OPENAI_API_KEY is injected as a HF Space secret (env var)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
