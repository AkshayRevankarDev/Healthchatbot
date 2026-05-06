---
title: AarogyaVaani
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Mental Health Screening Agent — Milestone 2
**CSE 635 | University at Buffalo**

Conversational AI system that replaces rigid PHQ-9 questionnaires with empathetic adaptive dialogue. Infers PHQ-9 item scores (0–3) from natural patient conversation using OpenAI (gpt-4o) and local translation models.

---

## Architecture

```
mental_health_screening/
├── data/
│   └── dsm_kb.json          # DSM-5-TR knowledge base (5 PHQ-9 domains)
├── docs/
│   └── pipeline_diagram.png # System architecture diagram
├── reports/                 # Academic reports and LaTeX source files
├── frontend/                # React landing page and chat UI
├── generate_sessions.py     # Synthetic patient session generator
├── safety_monitor.py        # SBERT-based crisis detection
├── dialogue_manager.py      # LangGraph adaptive dialogue graph
├── inference_engine.py      # Chain-of-Thought PHQ-9 scoring
├── translator.py            # IndicTrans2 + Whisper multilingual support
├── evaluate.py              # 6-metric evaluation pipeline
├── app.py                   # Streamlit chat UI (legacy/debug)
├── server.py                # FastAPI backend API
├── start.sh                 # Launcher for React + FastAPI
├── sessions/                # Generated patient sessions (JSON)
├── figures/                 # Output PNG figures
└── logs/                    # Session audit logs
```

### Key Components

| Component | Technology | Role |
|-----------|-----------|------|
| LLM | OpenAI (gpt-4o-mini, gpt-4o) | Dialogue generation, CoT scoring |
| Translation | IndicTrans2, Whisper (local) | Real-time multilingual text/voice |
| Safety Monitor | SBERT all-MiniLM-L6-v2 | Real-time crisis detection (threshold 0.82) |
| Dialogue Graph | LangGraph StateGraph | Adaptive conversation flow |
| UI | React + FastAPI | Web application interface |

---

## Setup

### Prerequisites

```bash
# 1. Configure OpenAI API Key
# Create a .env file in the project root:
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# 2. Install Python Dependencies
pip install -r requirements.txt

# 3. Download Translation Models (one-time ~800MB download)
python download_indictrans2.py
```

Python 3.10+ required. Requires `npm` for the frontend.

---

## Usage

### Step 1: Generate Synthetic Sessions

```bash
cd mental_health_screening
python generate_sessions.py
```

Generates 10 patient sessions (3 minimal, 3 mild, 2 moderate, 2 severe) in `sessions/`.

### Step 2: Run Evaluation

```bash
python evaluate.py
```

Runs all sessions through the dialogue manager and computes 6 metrics:
- Cohen's weighted kappa
- Disclosure efficiency (mean turns to confidence threshold)
- Safety recall and precision
- Discourse effectiveness (% domains scored)
- Mean response latency
- RAGAS faithfulness

Results saved to `results.json`.

### Step 3: Generate Figures

```bash
python visualize_results.py
```

Saves 3 figures to `figures/`:
- `scores_comparison.png` — Predicted vs gold PHQ-9 scores by domain
- `confidence_progression.png` — Confidence over turns for 3 sessions
- `metrics_radar.png` — All 6 metrics vs targets (radar chart)

### Step 4: Launch Web UI

```bash
bash start.sh
```

Opens the React landing page at http://localhost:3000 with:
- Web-based landing page and chat interface
- Voice recording and multilingual text input
- Real-time translation and safety monitoring

---

## PHQ-9 Domains Covered

| Domain | PHQ-9 Item | Description |
|--------|-----------|-------------|
| `mood` | Item 1 | Depressed mood / anhedonia |
| `sleep` | Item 3 | Sleep disturbance |
| `energy` | Item 4 | Fatigue / loss of energy |
| `appetite` | Item 5 | Poor appetite or overeating |
| `concentration` | Item 7 | Concentration difficulty |

---

## System Design

### Dialogue Flow

```
Patient message
    → Safety check (SBERT)
    → Rapport phase (turns 1-3): keyword scanning + confidence bootstrapping
    → Screening phase: highest-confidence domain first
        → Probe questions until confidence ≥ 0.75 AND ≥ 2 turns
        → CoT scoring (Extract → Reason → Score)
    → Cycle remaining domains
    → Session complete when all 5 domains scored
```

### Confidence Model
- Starts at 0.0 for all domains
- Boosted +0.15 per keyword hit during rapport
- Estimated 0.0–0.4 increment per turn via OpenAI
- Never decreases
- Scoring triggered at ≥ 0.75 AND ≥ 2 probe turns

### Safety Monitor
- 37 high-risk phrases covering suicidal ideation, self-harm, passive ideation, burden ideation
- SBERT cosine similarity threshold: 0.82
- Crisis response with 988 Lifeline reference

---

## Multilingual Support

Uses `IndicTrans2` (offline) for translating Indian languages to English, and `openai-whisper` (offline) for audio transcription. Ensures LLM processing is always done in English for highest accuracy, while the patient interacts in their native language.

---

## Safety Notice

This tool is for **research and educational purposes only**. It is not a clinical diagnostic instrument and must not be used as a substitute for professional mental health assessment. If you or someone you know is in crisis, call or text **988** (Suicide & Crisis Lifeline).
