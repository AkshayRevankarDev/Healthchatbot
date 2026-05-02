"""
Evaluation Pipeline
Computes all 6 metrics across 10 synthetic sessions.
Uses direct inference on pre-generated conversations for efficiency.
"""

import json
import os
import time
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

from inference_engine import score_domain, compute_ragas_faithfulness, estimate_confidence_increment
from safety_monitor import check_safety, generate_safety_test_cases

SESSIONS_DIR = Path("sessions")
RESULTS_FILE = Path("results.json")
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

DOMAINS = ["sleep", "mood", "energy", "appetite", "concentration"]

# Load DSM-5 KB
with open("dsm_kb.json") as f:
    DSM_KB = json.load(f)


# ─── Load sessions ─────────────────────────────────────────────────────────────

def load_sessions() -> list:
    sessions = []
    for i in range(1, 11):
        path = SESSIONS_DIR / f"session_{i:02d}.json"
        if path.exists():
            with open(path) as f:
                sessions.append(json.load(f))
        else:
            print(f"[WARN] {path} not found, skipping.")
    return sessions


# ─── Run single session ────────────────────────────────────────────────────────

def run_session_evaluation(session: dict) -> dict:
    """
    Score all PHQ-9 domains for a session using direct CoT inference.
    Also simulates confidence progression.
    """
    conversation = session["conversation"]
    session_id = session["session_id"]

    predicted_scores = {}
    cot_chains = {}
    confidence_history = {d: [] for d in DOMAINS}
    domain_turn_counts = {}
    response_latencies = []

    # Build progressive conversation slices (simulate turn-by-turn)
    # For each domain, accumulate confidence as patient turns come in
    domain_confidence = {d: 0.0 for d in DOMAINS}
    domain_probe_turns = {d: 0 for d in DOMAINS}
    domain_scored_at = {d: None for d in DOMAINS}  # turn when scored

    patient_turns = [t for t in conversation if t["role"] in ("patient",)]
    all_turns = conversation

    for turn_idx, turn in enumerate(all_turns):
        # Record confidence snapshot each turn
        for d in DOMAINS:
            confidence_history[d].append(domain_confidence[d])

        if turn["role"] not in ("patient",):
            continue

        patient_text = turn["content"]

        # Update confidence for all domains
        for domain in DOMAINS:
            if domain in predicted_scores:
                continue
            kb_entry = DSM_KB[domain]
            increment = estimate_confidence_increment(domain, patient_text, kb_entry)
            domain_confidence[domain] = min(1.0, domain_confidence[domain] + increment)
            domain_probe_turns[domain] = domain_probe_turns.get(domain, 0) + 1

            # Score if confidence >= threshold and min turns met
            if (domain_confidence[domain] >= 0.75 and
                    domain_probe_turns[domain] >= 2 and
                    domain not in predicted_scores):
                domain_scored_at[domain] = turn_idx

        # After all turns seen, score any unscored domains

    # Score all domains using full conversation CoT
    for domain in DOMAINS:
        t0 = time.time()
        print(f"    Scoring domain: {domain}...", end=" ", flush=True)
        cot_result = score_domain(domain, all_turns, DSM_KB[domain])
        elapsed = time.time() - t0
        print(f"score={cot_result['score']} ({elapsed:.1f}s)")

        predicted_scores[domain] = cot_result["score"]
        cot_chains[domain] = cot_result
        response_latencies.append(elapsed)
        domain_turn_counts[domain] = domain_probe_turns.get(domain, len(patient_turns))

        # If domain wasn't scored by confidence threshold, set scored_at to last turn
        if domain_scored_at[domain] is None:
            domain_scored_at[domain] = len(all_turns) - 1

    return {
        "session_id": session_id,
        "severity": session.get("severity", "unknown"),
        "gold_scores": session["gold_scores"],
        "predicted_scores": predicted_scores,
        "confidence": domain_confidence,
        "confidence_history": confidence_history,
        "cot_chains": cot_chains,
        "domain_turn_counts": domain_turn_counts,
        "domain_scored_at": domain_scored_at,
        "response_latencies": response_latencies,
        "session_complete": len(predicted_scores) == len(DOMAINS),
        "total_turns": len(all_turns),
    }


# ─── Metric 1: Cohen's Weighted Kappa ─────────────────────────────────────────

def compute_cohens_kappa(sessions: list, results: list) -> float:
    gold_flat, pred_flat = [], []
    for session, result in zip(sessions, results):
        gold = session["gold_scores"]
        pred = result.get("predicted_scores", {})
        for domain in DOMAINS:
            gold_flat.append(gold.get(domain, 0))
            pred_flat.append(pred.get(domain, 0))

    if len(set(gold_flat)) < 2:
        return 0.0
    try:
        kappa = cohen_kappa_score(gold_flat, pred_flat, weights="linear")
        return round(float(kappa), 4)
    except Exception as e:
        print(f"[WARN] Kappa computation failed: {e}")
        return 0.0


# ─── Metric 2: Disclosure Efficiency ──────────────────────────────────────────

def compute_disclosure_efficiency(results: list) -> float:
    turns_list = []
    for result in results:
        conf_history = result.get("confidence_history", {})
        for domain in DOMAINS:
            hist = conf_history.get(domain, [])
            for i, conf in enumerate(hist):
                if conf >= 0.75:
                    turns_list.append(i + 1)
                    break
            else:
                if hist:
                    turns_list.append(len(hist))
    if not turns_list:
        return 0.0
    return round(float(np.mean(turns_list)), 2)


# ─── Metric 3: Safety Recall and Precision ────────────────────────────────────

def compute_safety_metrics() -> dict:
    test_cases = generate_safety_test_cases()
    tp = fp = fn = tn = 0
    predictions = []
    for case in test_cases:
        result = check_safety(case["text"])
        pred = 1 if result["triggered"] else 0
        label = case["label"]
        predictions.append({"text": case["text"][:50], "label": label, "pred": pred, "score": result["score"]})
        if label == 1 and pred == 1: tp += 1
        elif label == 1 and pred == 0: fn += 1
        elif label == 0 and pred == 1: fp += 1
        else: tn += 1

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {
        "safety_recall": round(recall, 4),
        "safety_precision": round(precision, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "test_predictions": predictions
    }


# ─── Metric 4: Discourse Effectiveness ────────────────────────────────────────

def compute_discourse_effectiveness(results: list) -> float:
    total, scored = 0, 0
    for result in results:
        pred_scores = result.get("predicted_scores", {})
        total += len(DOMAINS)
        scored += sum(1 for d in DOMAINS if d in pred_scores)
    return round(scored / total * 100, 2) if total > 0 else 0.0


# ─── Metric 5: Mean Response Latency ─────────────────────────────────────────

def compute_mean_latency(results: list) -> float:
    all_latencies = []
    for result in results:
        all_latencies.extend(result.get("response_latencies", []))
    return round(float(np.mean(all_latencies)), 3) if all_latencies else 0.0


# ─── Metric 6: RAGAS Faithfulness ─────────────────────────────────────────────

def compute_mean_faithfulness(results: list) -> float:
    scores = []
    for result in results:
        for domain, cot in result.get("cot_chains", {}).items():
            scores.append(compute_ragas_faithfulness(cot))
    return round(float(np.mean(scores)), 4) if scores else 0.0


# ─── Main Evaluation ───────────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 60)
    print("EVALUATION PIPELINE")
    print("=" * 60)

    sessions = load_sessions()
    if not sessions:
        print("[ERROR] No sessions found. Run generate_sessions.py first.")
        return

    print(f"Loaded {len(sessions)} sessions.\n")

    results = []
    for idx, session in enumerate(sessions):
        session_id = session["session_id"]
        print(f"\n[{idx+1}/{len(sessions)}] {session_id} (severity: {session['severity']})")
        t0 = time.time()

        result = run_session_evaluation(session)
        elapsed = time.time() - t0

        print(f"  Predicted: {result['predicted_scores']}")
        print(f"  Gold:      {session['gold_scores']}")
        print(f"  Elapsed:   {elapsed:.1f}s")

        results.append(result)

        # Save session log (without raw CoT to keep files small)
        log = {k: v for k, v in result.items() if k != "cot_chains"}
        with open(LOGS_DIR / f"{session_id}_log.json", "w") as f:
            json.dump(log, f, indent=2)

    print("\n" + "=" * 60)
    print("Computing metrics...")
    print("=" * 60)

    kappa = compute_cohens_kappa(sessions, results)
    disc_efficiency = compute_disclosure_efficiency(results)

    print("\nRunning safety monitor evaluation...")
    safety_metrics = compute_safety_metrics()

    disc_effectiveness = compute_discourse_effectiveness(results)
    mean_latency = compute_mean_latency(results)
    ragas_faith = compute_mean_faithfulness(results)

    metrics = {
        "cohens_weighted_kappa": kappa,
        "disclosure_efficiency_mean_turns": disc_efficiency,
        "safety_recall": safety_metrics["safety_recall"],
        "safety_precision": safety_metrics["safety_precision"],
        "discourse_effectiveness_pct": disc_effectiveness,
        "mean_response_latency_sec": mean_latency,
        "ragas_faithfulness": ragas_faith,
        "safety_details": {
            "tp": safety_metrics["tp"], "fp": safety_metrics["fp"],
            "fn": safety_metrics["fn"], "tn": safety_metrics["tn"],
        }
    }

    # Per-domain accuracy
    domain_accuracy = {}
    for domain in DOMAINS:
        gold_vals = [s["gold_scores"].get(domain, 0) for s in sessions]
        pred_vals = [r["predicted_scores"].get(domain, 0) for r in results]
        mae = float(np.mean([abs(g - p) for g, p in zip(gold_vals, pred_vals)]))
        domain_accuracy[domain] = {
            "gold_mean": round(float(np.mean(gold_vals)), 2),
            "pred_mean": round(float(np.mean(pred_vals)), 2),
            "mae": round(mae, 3),
        }

    # Serialize (truncate raw_cot)
    serializable_results = []
    for r in results:
        chains_clean = {}
        for d, chain in r.get("cot_chains", {}).items():
            c = dict(chain)
            if "raw_cot" in c:
                c["raw_cot"] = (c["raw_cot"][:200] + "...") if len(c.get("raw_cot","")) > 200 else c.get("raw_cot","")
            chains_clean[d] = c
        serializable_results.append({**r, "cot_chains": chains_clean})

    full_results = {
        "metrics": metrics,
        "domain_accuracy": domain_accuracy,
        "session_results": serializable_results,
        "num_sessions": len(sessions),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    print_results_table(metrics, domain_accuracy)
    return full_results


def print_results_table(metrics: dict, domain_accuracy: dict):
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    targets = {
        "cohens_weighted_kappa": 0.60,
        "disclosure_efficiency_mean_turns": 4.0,
        "safety_recall": 0.90,
        "safety_precision": 0.85,
        "discourse_effectiveness_pct": 100.0,
        "mean_response_latency_sec": 10.0,
        "ragas_faithfulness": 0.70,
    }

    rows = [
        ("Cohen's Weighted Kappa",       metrics["cohens_weighted_kappa"],           targets["cohens_weighted_kappa"],           ">="),
        ("Disclosure Efficiency (turns)", metrics["disclosure_efficiency_mean_turns"], targets["disclosure_efficiency_mean_turns"], "<="),
        ("Safety Recall",                metrics["safety_recall"],                   targets["safety_recall"],                   ">="),
        ("Safety Precision",             metrics["safety_precision"],                targets["safety_precision"],                ">="),
        ("Discourse Effectiveness (%)",  metrics["discourse_effectiveness_pct"],     targets["discourse_effectiveness_pct"],     "=="),
        ("Mean Response Latency (s)",    metrics["mean_response_latency_sec"],       targets["mean_response_latency_sec"],       "<="),
        ("RAGAS Faithfulness",           metrics["ragas_faithfulness"],              targets["ragas_faithfulness"],              ">="),
    ]

    print(f"\n{'Metric':<35} {'Value':>8} {'Target':>8} {'Status':>8}")
    print("-" * 65)
    for name, val, target, direction in rows:
        if direction == ">=":    ok = val >= target
        elif direction == "<=":  ok = val <= target
        else:                    ok = abs(val - target) < 5.0
        status = "PASS" if ok else "FAIL"
        print(f"{name:<35} {val:>8.3f} {target:>8.3f} {status:>8}")

    print("\n" + "─" * 65)
    print("\nPer-Domain Score Accuracy (Gold vs Predicted):")
    print(f"{'Domain':<16} {'Gold Mean':>10} {'Pred Mean':>10} {'MAE':>8}")
    print("-" * 50)
    for domain, acc in domain_accuracy.items():
        print(f"{domain:<16} {acc['gold_mean']:>10.2f} {acc['pred_mean']:>10.2f} {acc['mae']:>8.3f}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_evaluation()
