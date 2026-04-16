# Mental Health Screening Agent — Milestone 2
**CSE 635 | University at Buffalo**

Conversational AI system that replaces rigid PHQ-9 questionnaires with empathetic adaptive dialogue. Infers PHQ-9 item scores (0–3) from natural patient conversation using Ollama llama3 locally.

---

## Architecture

```
mental_health_screening/
├── dsm_kb.json              # DSM-5-TR knowledge base (5 PHQ-9 domains)
├── generate_sessions.py     # Synthetic patient session generator
├── safety_monitor.py        # SBERT-based crisis detection
├── dialogue_manager.py      # LangGraph adaptive dialogue graph
├── inference_engine.py      # Chain-of-Thought PHQ-9 scoring
├── evaluate.py              # 6-metric evaluation pipeline
├── app.py                   # Streamlit chat UI
├── visualize_results.py     # Matplotlib result figures
├── sessions/                # Generated patient sessions (JSON)
├── figures/                 # Output PNG figures
└── logs/                    # Session audit logs
```

### Key Components

| Component | Technology | Role |
|-----------|-----------|------|
| LLM | Ollama llama3 (local) | Dialogue generation, CoT scoring |
| Safety Monitor | SBERT all-MiniLM-L6-v2 | Real-time crisis detection (threshold 0.82) |
| Dialogue Graph | LangGraph StateGraph | Adaptive conversation flow |
| CoT Engine | Ollama llama3 | 3-step DSM-5-TR grounded scoring |
| UI | Streamlit | Live confidence bars, chat interface |

---

## Setup

### Prerequisites

```bash
# 1. Install Ollama (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh   # Linux/Mac
# Windows: download installer from ollama.ai

# 2. Pull llama3 (~4GB one-time download)
ollama pull llama3

# 3. Verify
ollama run llama3 "say hello"
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

Python 3.10+ required.

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

### Step 4: Launch Streamlit UI

```bash
streamlit run app.py
```

Opens at http://localhost:8501 with:
- Left panel: chat interface
- Right panel: live confidence bars + final scores
- Red safety banner if crisis content detected

---

## PHQ-9 Domains Covered

| Domain | PHQ-9 Item | Description |
|--------|-----------|-------------|
| `mood` | Item 1 | Depressed mood / anhedonia |
| `sleep` | Item 3 | Sleep disturbance |
| `energy` | Item 4 | Fatigue / loss of energy |
| `self_worth` | Item 6 | Worthlessness / guilt |
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
- Estimated 0.0–0.4 increment per turn via Ollama
- Never decreases
- Scoring triggered at ≥ 0.75 AND ≥ 2 probe turns

### Safety Monitor
- 37 high-risk phrases covering suicidal ideation, self-harm, passive ideation, burden ideation
- SBERT cosine similarity threshold: 0.82
- Crisis response with 988 Lifeline reference

---

## Fully Offline

After `ollama pull llama3` and `pip install -r requirements.txt`, the entire system runs offline. No API keys or internet connection required.

---

## Safety Notice

This tool is for **research and educational purposes only**. It is not a clinical diagnostic instrument and must not be used as a substitute for professional mental health assessment. If you or someone you know is in crisis, call or text **988** (Suicide & Crisis Lifeline).
