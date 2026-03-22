# analysis/oracle_dashboard.py
"""
Oracle Dashboard — Week 9

Generates four matplotlib plots from the SQLite database showing
generation-over-generation behaviour of the TRI_FLAG + ACEGEN loop:

    1. Reward distribution per generation (box plot)
    2. Pass/Flag/Discard rate trends (stacked bar)
    3. Mean reward trajectory (line chart, generation on x-axis)
    4. Scaffold diversity per generation (bar chart)

Usage (CLI):
    python -m analysis.oracle_dashboard --db runs/triflag.db --output-dir analysis/output/oracle

Usage (programmatic):
    from analysis.oracle_dashboard import generate_dashboard
    generate_dashboard(db_path="runs/triflag.db")
"""

import argparse
import os
import sqlite3
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from analysis.diversity import compute_diversity

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for server / CI use
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_runs(
    db_path: str,
    batch_ids: Optional[List[str]] = None,
) -> List[Dict]:
    """Load triage run rows from SQLite, optionally filtered by batch_ids."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if batch_ids:
            placeholders = ",".join("?" * len(batch_ids))
            rows = conn.execute(
                f"SELECT * FROM triage_runs WHERE batch_id IN ({placeholders})",
                batch_ids,
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM triage_runs").fetchall()
    finally:
        conn.close()

    return [dict(row) for row in rows]


def _group_by_generation(runs: List[Dict]) -> Dict[str, List[Dict]]:
    """Group runs by batch_id (generation label). NULL batch_id → 'ungrouped'."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for run in runs:
        key = run.get("batch_id") or "ungrouped"
        groups[key].append(run)
    return dict(groups)


def _generation_summary(
    generations: Dict[str, List[Dict]],
) -> List[Dict]:
    """Compute per-generation summary stats for the CSV and plots."""
    summaries = []
    for batch_id, runs in sorted(generations.items()):
        rewards = [r["reward"] for r in runs if r.get("reward") is not None]
        decisions = [r.get("final_decision", "") for r in runs]
        scaffolds = [r.get("scaffold_smiles") for r in runs]

        total = len(runs)
        diversity = compute_diversity(scaffolds)
        pass_count = decisions.count("PASS")
        flag_count = decisions.count("FLAG")
        discard_count = decisions.count("DISCARD")

        summaries.append({
            "batch_id": batch_id,
            "count": total,
            "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
            "std_reward": round(_std(rewards), 4) if len(rewards) > 1 else 0.0,
            "pass_rate": round(pass_count / total, 4) if total else 0.0,
            "flag_rate": round(flag_count / total, 4) if total else 0.0,
            "discard_rate": round(discard_count / total, 4) if total else 0.0,
            "unique_scaffolds": diversity.unique_scaffolds,
            "convergence_warning": diversity.convergence_warning,
            "rewards_raw": rewards,
        })
    return summaries


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance ** 0.5


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_reward_distribution(
    summaries: List[Dict],
    output_path: str,
) -> None:
    """Box plot: reward distribution per generation."""
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [s["batch_id"] for s in summaries]
    data = [s["rewards_raw"] for s in summaries]

    # Filter to generations with at least one reward
    filtered = [(l, d) for l, d in zip(labels, data) if d]
    if not filtered:
        plt.close(fig)
        return
    labels_f, data_f = zip(*filtered)

    ax.boxplot(data_f, labels=labels_f, patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.7))
    ax.set_title("Reward Distribution per Generation", fontsize=13, fontweight="bold")
    ax.set_xlabel("Generation (batch_id)")
    ax.set_ylabel("Reward [0, 1]")
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_decision_rates(
    summaries: List[Dict],
    output_path: str,
) -> None:
    """Stacked bar: PASS / FLAG / DISCARD rates per generation."""
    labels = [s["batch_id"] for s in summaries]
    pass_rates = [s["pass_rate"] for s in summaries]
    flag_rates = [s["flag_rate"] for s in summaries]
    discard_rates = [s["discard_rate"] for s in summaries]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, pass_rates, label="PASS", color="#2ca02c", alpha=0.85)
    ax.bar(x, flag_rates, bottom=pass_rates, label="FLAG", color="#ff7f0e", alpha=0.85)
    bottoms = [p + f for p, f in zip(pass_rates, flag_rates)]
    ax.bar(x, discard_rates, bottom=bottoms, label="DISCARD", color="#d62728", alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Decision Rates per Generation", fontsize=13, fontweight="bold")
    ax.set_ylabel("Fraction of batch")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_mean_reward_trajectory(
    summaries: List[Dict],
    output_path: str,
) -> None:
    """Line chart: mean reward trajectory across generations."""
    labels = [s["batch_id"] for s in summaries]
    means = [s["mean_reward"] for s in summaries]
    stds = [s["std_reward"] for s in summaries]
    x = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, means, marker="o", linewidth=2, color="#4C72B0", label="Mean reward")
    ax.fill_between(
        x,
        [max(0, m - s) for m, s in zip(means, stds)],
        [min(1, m + s) for m, s in zip(means, stds)],
        alpha=0.2, color="#4C72B0", label="±1 std",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Mean Reward Trajectory", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Reward")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_scaffold_diversity(
    summaries: List[Dict],
    output_path: str,
) -> None:
    """Bar chart: unique scaffold count per generation."""
    labels = [s["batch_id"] for s in summaries]
    unique = [s["unique_scaffolds"] for s in summaries]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(x, unique, color="#9467bd", alpha=0.8)
    # Convergence warning overlay
    for i, s in enumerate(summaries):
        if s["convergence_warning"]:
            bars[i].set_edgecolor("red")
            bars[i].set_linewidth(2.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(
        "Unique Scaffolds per Generation  (red border = mode collapse warning)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Unique scaffolds")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def _write_summary_csv(summaries: List[Dict], output_path: str) -> None:
    import csv
    fieldnames = [
        "batch_id", "count", "mean_reward", "std_reward",
        "pass_rate", "flag_rate", "discard_rate",
        "unique_scaffolds", "convergence_warning",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s[k] for k in fieldnames})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dashboard(
    db_path: str = "runs/triflag.db",
    batch_ids: Optional[List[str]] = None,
    output_dir: str = "analysis/output/oracle",
) -> List[Dict]:
    """
    Generate the four oracle dashboard plots and summary CSV.

    Args:
        db_path:    Path to triflag.db.
        batch_ids:  Optional list of batch_ids to include. None = all.
        output_dir: Directory for .png and .csv outputs.

    Returns:
        List of per-generation summary dicts (same data written to CSV).
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for oracle_dashboard. "
            "Install with: pip install matplotlib"
        )

    os.makedirs(output_dir, exist_ok=True)

    runs = _load_runs(db_path, batch_ids)
    if not runs:
        print(f"No runs found in {db_path} — dashboard not generated.")
        return []

    generations = _group_by_generation(runs)
    summaries = _generation_summary(generations)

    _plot_reward_distribution(summaries, os.path.join(output_dir, "reward_distribution.png"))
    _plot_decision_rates(summaries, os.path.join(output_dir, "decision_rates.png"))
    _plot_mean_reward_trajectory(summaries, os.path.join(output_dir, "mean_reward_trajectory.png"))
    _plot_scaffold_diversity(summaries, os.path.join(output_dir, "scaffold_diversity.png"))
    _write_summary_csv(summaries, os.path.join(output_dir, "generation_summary.csv"))

    print(f"Dashboard written to {output_dir}/")
    return summaries


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRI_FLAG Oracle Dashboard")
    parser.add_argument("--db", default="runs/triflag.db", help="Path to triflag.db")
    parser.add_argument(
        "--batch-ids", nargs="*", default=None,
        help="Space-separated batch IDs to include (default: all)",
    )
    parser.add_argument(
        "--output-dir", default="analysis/output/oracle",
        help="Output directory for plots and CSV",
    )
    args = parser.parse_args()
    generate_dashboard(
        db_path=args.db,
        batch_ids=args.batch_ids,
        output_dir=args.output_dir,
    )