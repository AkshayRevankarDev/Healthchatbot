"""
Results Visualization
Generates 3 matplotlib figures from results.json.
"""

import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_FILE = Path("results.json")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

DOMAINS = ["sleep", "mood", "energy", "appetite", "concentration"]
DOMAIN_LABELS = ["Sleep", "Mood", "Energy", "Appetite", "Concentration"]

DOMAIN_COLORS = {
    "sleep": "#4e79a7",
    "mood": "#f28e2b",
    "energy": "#59a14f",
    "appetite": "#b07aa1",
    "concentration": "#e15759",
}


def load_results() -> dict:
    with open(RESULTS_FILE) as f:
        return json.load(f)


# ─── Figure 1: Predicted vs Gold Scores (Bar Chart) ───────────────────────────

def figure1_scores_comparison(results: dict):
    """Bar chart of predicted vs gold scores per domain (averaged across sessions)."""
    domain_acc = results.get("domain_accuracy", {})

    gold_means = [domain_acc.get(d, {}).get("gold_mean", 0) for d in DOMAINS]
    pred_means = [domain_acc.get(d, {}).get("pred_mean", 0) for d in DOMAINS]
    maes = [domain_acc.get(d, {}).get("mae", 0) for d in DOMAINS]

    x = np.arange(len(DOMAINS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, gold_means, width, label="Gold (True)", color="#2196F3", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, pred_means, width, label="Predicted", color="#FF9800", alpha=0.85, edgecolor="white")

    # Add MAE annotations
    for i, (b1, b2, mae) in enumerate(zip(bars1, bars2, maes)):
        ax.annotate(
            f"MAE={mae:.2f}",
            xy=(x[i], max(gold_means[i], pred_means[i]) + 0.05),
            ha="center", va="bottom",
            fontsize=9, color="#555"
        )

    ax.set_xlabel("PHQ-9 Domain", fontsize=12)
    ax.set_ylabel("Mean Score (0–3)", fontsize=12)
    ax.set_title("Figure 1: Predicted vs Gold PHQ-9 Scores by Domain\n(Averaged Across 10 Sessions)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(DOMAIN_LABELS, fontsize=11)
    ax.set_ylim(0, 3.5)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Add overall kappa
    kappa = results.get("metrics", {}).get("cohens_weighted_kappa", 0)
    ax.text(0.02, 0.97, f"Cohen's κ = {kappa:.3f}", transform=ax.transAxes,
            fontsize=10, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8))

    plt.tight_layout()
    out = FIGURES_DIR / "scores_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─── Figure 2: Confidence Progression (Line Chart) ────────────────────────────

def figure2_confidence_progression(results: dict):
    """Line chart of confidence over turns for 3 representative sessions."""
    sessions = results.get("session_results", [])
    if not sessions:
        print("[WARN] No session results for Figure 2.")
        return

    # Pick 3 representative sessions: minimal, moderate, severe
    target_severities = ["minimal", "moderate", "severe"]
    selected = []
    for sev in target_severities:
        for s in sessions:
            if s.get("severity") == sev:
                selected.append(s)
                break
    # Fallback: just take first 3
    if len(selected) < 3:
        selected = sessions[:3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, session in zip(axes, selected):
        conf_hist = session.get("confidence_history", {})
        session_id = session["session_id"]
        severity = session.get("severity", "unknown")
        gold = session.get("gold_scores", {})
        total_gold = sum(gold.values())

        for domain in DOMAINS:
            hist = conf_hist.get(domain, [])
            if hist:
                turns = list(range(1, len(hist) + 1))
                ax.plot(turns, hist,
                        color=DOMAIN_COLORS[domain],
                        marker="o", markersize=3,
                        linewidth=2,
                        label=domain.replace("_", " ").title())

        # Threshold line
        ax.axhline(y=0.75, color="red", linestyle="--", alpha=0.6, linewidth=1.5, label="Threshold (0.75)")

        ax.set_title(f"{session_id}\nSeverity: {severity.title()} (PHQ-9={total_gold})",
                    fontsize=10, fontweight="bold")
        ax.set_xlabel("Turn Number", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    axes[0].set_ylabel("Confidence Score", fontsize=11)
    axes[1].set_title(axes[1].get_title())

    # Shared legend
    handles = [mpatches.Patch(color=DOMAIN_COLORS[d], label=d.replace("_", " ").title()) for d in DOMAINS]
    handles.append(plt.Line2D([0], [0], color="red", linestyle="--", label="Threshold (0.75)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("Figure 2: Domain Confidence Progression Over Conversation Turns",
                fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    out = FIGURES_DIR / "confidence_progression.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─── Figure 3: Metrics Radar Chart ────────────────────────────────────────────

def figure3_metrics_radar(results: dict):
    """Spider/radar chart of all 6 metrics vs targets."""
    metrics = results.get("metrics", {})

    # Normalize values to 0-1 range for radar
    metric_specs = [
        {
            "name": "Cohen's κ\n(target≥0.60)",
            "value": metrics.get("cohens_weighted_kappa", 0),
            "target": 0.60,
            "max": 1.0,
            "min": -1.0,
            "higher_better": True,
        },
        {
            "name": "Safety\nRecall\n(target≥0.90)",
            "value": metrics.get("safety_recall", 0),
            "target": 0.90,
            "max": 1.0,
            "min": 0.0,
            "higher_better": True,
        },
        {
            "name": "Safety\nPrecision\n(target≥0.85)",
            "value": metrics.get("safety_precision", 0),
            "target": 0.85,
            "max": 1.0,
            "min": 0.0,
            "higher_better": True,
        },
        {
            "name": "RAGAS\nFaithfulness\n(target≥0.70)",
            "value": metrics.get("ragas_faithfulness", 0),
            "target": 0.70,
            "max": 1.0,
            "min": 0.0,
            "higher_better": True,
        },
        {
            "name": "Discourse\nEffectiveness\n(target=100%)",
            "value": metrics.get("discourse_effectiveness_pct", 0) / 100.0,
            "target": 1.0,
            "max": 1.0,
            "min": 0.0,
            "higher_better": True,
        },
        {
            "name": "Efficiency\n(turns≤4)\nnormalized",
            "value": max(0, 1.0 - (metrics.get("disclosure_efficiency_mean_turns", 8) - 1) / 7),
            "target": max(0, 1.0 - (4 - 1) / 7),
            "max": 1.0,
            "min": 0.0,
            "higher_better": True,
        },
    ]

    labels = [m["name"] for m in metric_specs]
    values = [m["value"] for m in metric_specs]
    targets = [m["target"] for m in metric_specs]

    N = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    values_plot = values + values[:1]
    targets_plot = targets + targets[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Grid
    ax.set_facecolor("#fafafa")
    for level in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles, [level] * (N + 1), color="lightgray", linewidth=0.8, linestyle=":")

    # Target area
    ax.fill(angles, targets_plot, alpha=0.12, color="#2196F3")
    ax.plot(angles, targets_plot, color="#2196F3", linewidth=2, linestyle="--", label="Target", alpha=0.8)

    # Actual values
    ax.fill(angles, values_plot, alpha=0.25, color="#FF5722")
    ax.plot(angles, values_plot, color="#FF5722", linewidth=2.5, marker="o", markersize=8, label="Achieved")

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, color="#333")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="gray")

    # Value annotations
    for angle, val, label in zip(angles[:-1], values, labels):
        ax.annotate(
            f"{val:.2f}",
            xy=(angle, val),
            xytext=(angle, val + 0.07),
            fontsize=8, ha="center", color="#FF5722", fontweight="bold"
        )

    ax.set_title("Figure 3: System Performance Metrics vs Targets\n(Radar Chart)",
                fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=10)

    plt.tight_layout()
    out = FIGURES_DIR / "metrics_radar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("RESULTS VISUALIZATION")
    print("=" * 60)

    if not RESULTS_FILE.exists():
        print(f"[ERROR] {RESULTS_FILE} not found. Run evaluate.py first.")
        return

    results = load_results()
    print(f"Loaded results from {RESULTS_FILE}")
    print(f"Sessions: {results.get('num_sessions', 0)}")

    print("\nGenerating Figure 1: Scores Comparison...")
    figure1_scores_comparison(results)

    print("Generating Figure 2: Confidence Progression...")
    figure2_confidence_progression(results)

    print("Generating Figure 3: Metrics Radar...")
    figure3_metrics_radar(results)

    print("\n" + "=" * 60)
    print("All figures saved to figures/")
    print("  figures/scores_comparison.png")
    print("  figures/confidence_progression.png")
    print("  figures/metrics_radar.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
