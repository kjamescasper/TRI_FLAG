"""
analysis/sa_score_distribution.py

Week 4: SA Score Distribution Analysis for TRI_FLAG

Computes SA scores and complexity breakdowns for a batch of molecules.

Outputs:
  1. Console: summary statistics, per-molecule table, benchmark validation
  2. sa_score_histogram.png    — score distribution with threshold bands
  3. sa_score_pie_chart.png    — PASS / FLAG / DISCARD breakdown
  4. sa_score_complexity.png   — scatter: SA score vs. heavy atom count
  5. sa_score_results.csv      — per-molecule results table

Usage:
    # Built-in 30-molecule reference set:
    python analysis/sa_score_distribution.py

    # Custom CSV (columns: molecule_id, smiles[, name]):
    python analysis/sa_score_distribution.py --input data/my_molecules.csv

    # Validate against benchmark molecules:
    python analysis/sa_score_distribution.py --benchmarks

    # Use lead-optimization thresholds:
    python analysis/sa_score_distribution.py --context lead_opt

    # Save to custom directory:
    python analysis/sa_score_distribution.py --output-dir results/week4/
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Ensure triage_agent package importable when run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chemistry.sa_score import full_sa_analysis, validate_benchmarks
from policies.thresholds import (
    DEFAULT_SA_THRESHOLDS,
    LEAD_OPTIMIZATION_THRESHOLDS,
    NATURAL_PRODUCT_THRESHOLDS,
    FRAGMENT_SCREENING_THRESHOLDS,
    SAScoreThresholds,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _PLT = True
except ImportError:
    _PLT = False
    print("WARNING: matplotlib not found — plots skipped. "
          "Install with: pip install matplotlib")


# =============================================================================
# Reference molecule set (30 diverse drug-like + challenging structures)
# =============================================================================

REFERENCE_MOLECULES: List[Tuple[str, str, str]] = [
    # ── Easy ─────────────────────────────────────────────────────────────────
    ("REF_001", "CCO",                                       "Ethanol"),
    ("REF_002", "c1ccccc1",                                  "Benzene"),
    ("REF_003", "CC(=O)O",                                   "Acetic acid"),
    ("REF_004", "CC(N)C(=O)O",                               "Alanine"),
    ("REF_005", "c1ccc(N)cc1",                               "Aniline"),
    ("REF_006", "CC(=O)Nc1ccc(O)cc1",                       "Paracetamol"),
    ("REF_007", "CC(=O)Oc1ccccc1C(=O)O",                    "Aspirin"),
    ("REF_008", "OC(=O)c1ccccc1",                            "Benzoic acid"),
    ("REF_009", "Cc1ccc(S(N)(=O)=O)cc1",                    "Toluenesulfonamide"),
    ("REF_010", "O=C(O)c1ccc(Cl)cc1",                       "4-Chlorobenzoic acid"),
    # ── Moderate ─────────────────────────────────────────────────────────────
    ("REF_011", "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",      "Testosterone"),
    ("REF_012", "CN1CCC[C@H]1c1cccnc1",                     "Nicotine"),
    ("REF_013", "CC(C)Cc1ccc(C(C)C(=O)O)cc1",               "Ibuprofen"),
    ("REF_014", "OC(=O)[C@@H](N)Cc1ccc(O)cc1",              "Tyrosine"),
    ("REF_015", "Clc1ccc2c(c1)C(c1ccccc1)=NCC2",            "Benzodiazepine scaffold"),
    ("REF_016", "COc1cc2c(cc1OC)CC(N)CC2",                  "Dopamine analog"),
    ("REF_017", "Cn1c(=O)c2c(ncn2C)n(c1=O)C",              "Caffeine"),
    ("REF_018", "CC(=O)c1ccc(OCC(=O)O)cc1",                 "Fenofibrate analog"),
    ("REF_019", "Cc1nc2ccccc2c(=O)n1CC(=O)O",               "Zolpidem analog"),
    ("REF_020", "O=C1NC(=O)c2ccccc21",                      "Isatoic anhydride"),
    # ── Difficult / natural product-like ─────────────────────────────────────
    ("REF_021",
     "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",    "Cholesterol"),
    ("REF_022",
     "O=C(O[C@@H]1C[C@]2(O)C(=O)[C@H](OC(=O)c3ccccc3)[C@@H]2[C@@H]1OC(C)=O)c1ccccc1",
     "Taxol scaffold"),
    ("REF_023",
     "C[C@@H]1CC[C@H]2C[C@@H](/C(=C/[C@@H]3CC(=O)[C@H](C/C=C\\3)CC(C)=O)\\C)CC[C@@]2([C@H]1O)O",
     "Erythromycin scaffold"),
    ("REF_024", "O=C1c2ccccc2C(=O)c2c1cc1ccc3cccc4ccc2c1c34",  "Coronene diimide"),
    ("REF_025", "OC1OC(CO)C(O)C(O)C1O",                     "Glucose"),
    ("REF_026", "CC12CC3CC(CC(C3)C1)(C2)N",                  "Amantadine analog"),
    ("REF_027", "O=C(O)C1CC2CC1CC2=O",                       "Norbornane diacid"),
    ("REF_028", "C1CC2CCCC3CCCC1(CC23)C",                    "Twistane"),
    ("REF_029",
     "[C@@H]1(O)[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O", "Inositol"),
    ("REF_030", "O=C(O)c1cc(Cl)c(OCC(F)(F)F)c(Cl)c1",       "Triclopyr analog"),
]

THRESHOLD_SETS = {
    "default":    DEFAULT_SA_THRESHOLDS,
    "lead_opt":   LEAD_OPTIMIZATION_THRESHOLDS,
    "nat_prod":   NATURAL_PRODUCT_THRESHOLDS,
    "fragment":   FRAGMENT_SCREENING_THRESHOLDS,
}


# =============================================================================
# Data structure
# =============================================================================

@dataclass
class MoleculeResult:
    molecule_id: str
    smiles: str
    name: str
    sa_score: Optional[float]
    synthesizability_category: Optional[str]
    decision: str
    num_heavy_atoms: Optional[int]
    num_stereocenters: Optional[int]
    num_rings: Optional[int]
    warning_flags: List[str]
    error_message: Optional[str]


# =============================================================================
# Core analysis
# =============================================================================

def load_molecules_from_csv(filepath: str, smiles_col: str = "smiles") -> List[Tuple]:
    molecules = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            mol_id = row.get("molecule_id", f"MOL_{i+1:04d}")
            smiles = row.get(smiles_col, "").strip()
            name = row.get("name", mol_id)
            if smiles:
                molecules.append((mol_id, smiles, name))
    return molecules


def compute_batch(
    molecules: List[Tuple[str, str, str]],
    thresholds: SAScoreThresholds,
) -> List[MoleculeResult]:
    results = []
    for mol_id, smiles, name in molecules:
        analysis = full_sa_analysis(smiles)
        if analysis["success"]:
            score = analysis["sa_score"]
            bd = analysis["complexity_breakdown"]
            decision = thresholds.classify(score)
            category = thresholds.categorize(score)
            results.append(MoleculeResult(
                molecule_id=mol_id,
                smiles=smiles,
                name=name,
                sa_score=score,
                synthesizability_category=category,
                decision=decision,
                num_heavy_atoms=bd.get("num_heavy_atoms"),
                num_stereocenters=bd.get("num_stereocenters"),
                num_rings=bd.get("num_rings"),
                warning_flags=bd.get("warning_flags", []),
                error_message=None,
            ))
        else:
            results.append(MoleculeResult(
                molecule_id=mol_id, smiles=smiles, name=name,
                sa_score=None, synthesizability_category=None,
                decision="ERROR",
                num_heavy_atoms=None, num_stereocenters=None, num_rings=None,
                warning_flags=[], error_message=analysis["error_message"],
            ))
    return results


# =============================================================================
# Console output
# =============================================================================

def print_summary(results: List[MoleculeResult], thresholds: SAScoreThresholds) -> None:
    valid = [r for r in results if r.sa_score is not None]
    scores = [r.sa_score for r in valid]
    counts = {d: sum(1 for r in results if r.decision == d)
              for d in ("PASS", "FLAG", "DISCARD", "ERROR")}

    print("\n" + "=" * 66)
    print("  SA SCORE DISTRIBUTION ANALYSIS — TRI_FLAG Week 4")
    print("=" * 66)
    print(f"\n  Molecules total:  {len(results)}")
    print(f"  Scored:           {len(valid)}")
    print(f"  Errors:           {counts['ERROR']}")
    print()
    print(f"  Pipeline thresholds (current context):")
    print(f"    PASS    SA < {thresholds.pass_threshold}")
    print(f"    FLAG    SA {thresholds.pass_threshold} – {thresholds.flag_threshold}")
    print(f"    DISCARD SA > {thresholds.flag_threshold}")
    print()
    print(f"  Decisions:")
    n = len(results)
    print(f"    ✅ PASS    : {counts['PASS']:>4}  ({counts['PASS']/n*100:.1f}%)")
    print(f"    ⚠️  FLAG    : {counts['FLAG']:>4}  ({counts['FLAG']/n*100:.1f}%)")
    print(f"    ❌ DISCARD : {counts['DISCARD']:>4}  ({counts['DISCARD']/n*100:.1f}%)")
    if counts["ERROR"]:
        print(f"    💥 ERROR   : {counts['ERROR']:>4}  ({counts['ERROR']/n*100:.1f}%)")

    if scores:
        sorted_s = sorted(scores)
        mid = len(sorted_s) // 2
        median = sorted_s[mid] if len(sorted_s) % 2 else (sorted_s[mid-1] + sorted_s[mid]) / 2
        print()
        print(f"  Score statistics (n={len(scores)}):")
        print(f"    Min:    {min(scores):.2f}")
        print(f"    Max:    {max(scores):.2f}")
        print(f"    Mean:   {sum(scores)/len(scores):.2f}")
        print(f"    Median: {median:.2f}")

    print()
    print(f"  {'ID':<12} {'Score':>6}  {'Cat':<14}  {'Decision':<8}  Name")
    print(f"  {'-'*12} {'-'*6}  {'-'*14}  {'-'*8}  {'-'*28}")
    icons = {"PASS": "✅", "FLAG": "⚠️ ", "DISCARD": "❌", "ERROR": "💥"}
    for r in sorted(results, key=lambda x: (x.sa_score or 99)):
        score_str = f"{r.sa_score:.2f}" if r.sa_score is not None else "  N/A"
        cat = r.synthesizability_category or "—"
        icon = icons.get(r.decision, "  ")
        print(f"  {r.molecule_id:<12} {score_str:>6}  {cat:<14}  "
              f"{icon} {r.decision:<6}  {r.name}")
    print()


def print_benchmark_report() -> None:
    print("\n" + "=" * 66)
    print("  SA SCORE BENCHMARK VALIDATION")
    print("=" * 66)
    report = validate_benchmarks()
    total = report["passed"] + report["failed"]
    print(f"\n  Benchmarks: {total}   Passed: {report['passed']}   "
          f"Failed: {report['failed']}")
    print(f"  Pass rate: {report['passed']/total*100:.0f}%\n")
    for r in report["results"]:
        status = "✅" if r["status"] == "PASS" else "❌"
        score_str = f"{r['sa_score']:.2f}" if r["sa_score"] else " N/A"
        lo, hi = r["expected_range"]
        print(f"  {status} {r['name']:<22} score={score_str}  expected=[{lo:.1f},{hi:.1f}]")
    print()


# =============================================================================
# CSV export
# =============================================================================

def export_csv(results: List[MoleculeResult], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "molecule_id", "smiles", "name", "sa_score",
            "synthesizability_category", "decision",
            "num_heavy_atoms", "num_stereocenters", "num_rings",
            "num_warning_flags", "error_message",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "molecule_id": r.molecule_id,
                "smiles": r.smiles,
                "name": r.name,
                "sa_score": f"{r.sa_score:.4f}" if r.sa_score is not None else "",
                "synthesizability_category": r.synthesizability_category or "",
                "decision": r.decision,
                "num_heavy_atoms": r.num_heavy_atoms or "",
                "num_stereocenters": r.num_stereocenters if r.num_stereocenters is not None else "",
                "num_rings": r.num_rings if r.num_rings is not None else "",
                "num_warning_flags": len(r.warning_flags),
                "error_message": r.error_message or "",
            })
    print(f"  CSV exported to:  {output_path}")


# =============================================================================
# Plots
# =============================================================================

def plot_all(
    results: List[MoleculeResult],
    thresholds: SAScoreThresholds,
    output_dir: str,
) -> None:
    if not _PLT:
        return

    valid = [r for r in results if r.sa_score is not None]
    scores = [r.sa_score for r in valid]
    decisions = [r.decision for r in valid]

    if not scores:
        print("  No valid scores to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    colors = {"PASS": "#2ecc71", "FLAG": "#f39c12", "DISCARD": "#e74c3c"}

    # ------------------------------------------------------------------
    # 1. Histogram with threshold bands
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axvspan(1.0, thresholds.pass_threshold, alpha=0.07, color="#2ecc71")
    ax.axvspan(thresholds.pass_threshold, thresholds.flag_threshold, alpha=0.07, color="#f39c12")
    ax.axvspan(thresholds.flag_threshold, 10.0, alpha=0.07, color="#e74c3c")

    ax.axvline(thresholds.pass_threshold, color="#f39c12", linestyle="--", linewidth=1.5,
               label=f"FLAG threshold ({thresholds.pass_threshold})")
    ax.axvline(thresholds.flag_threshold, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"DISCARD threshold ({thresholds.flag_threshold})")

    n_bins = min(20, max(5, len(scores) // 2))
    ns, bins, patches = ax.hist(scores, bins=n_bins, range=(1.0, 10.0),
                                 edgecolor="white", linewidth=0.5)
    bin_width = bins[1] - bins[0]
    for patch, left_edge in zip(patches, bins[:-1]):
        mid = left_edge + bin_width / 2
        if mid < thresholds.pass_threshold:
            patch.set_facecolor("#2ecc71")
        elif mid <= thresholds.flag_threshold:
            patch.set_facecolor("#f39c12")
        else:
            patch.set_facecolor("#e74c3c")

    legend_patches = [
        mpatches.Patch(color="#2ecc71", label=f"PASS  (SA < {thresholds.pass_threshold})"),
        mpatches.Patch(color="#f39c12",
                       label=f"FLAG  ({thresholds.pass_threshold}–{thresholds.flag_threshold})"),
        mpatches.Patch(color="#e74c3c", label=f"DISCARD  (SA > {thresholds.flag_threshold})"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=10)
    ax.set_xlabel("SA Score (1 = easy, 10 = hard to synthesize)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("TRI_FLAG Week 4 — SA Score Distribution", fontsize=14, fontweight="bold")
    ax.set_xlim(1.0, 10.0)
    ax.grid(axis="y", alpha=0.3)

    hist_path = os.path.join(output_dir, "sa_score_histogram.png")
    fig.tight_layout()
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"  Histogram saved to:   {hist_path}")

    # ------------------------------------------------------------------
    # 2. Pie chart
    # ------------------------------------------------------------------
    counts = {d: decisions.count(d) for d in ("PASS", "FLAG", "DISCARD")}
    non_zero = {k: v for k, v in counts.items() if v > 0}
    if non_zero:
        fig2, ax2 = plt.subplots(figsize=(7, 7))
        ax2.pie(
            list(non_zero.values()),
            labels=list(non_zero.keys()),
            colors=[colors[k] for k in non_zero],
            autopct="%1.1f%%",
            startangle=140,
            textprops={"fontsize": 13},
        )
        ax2.set_title(
            f"TRI_FLAG Week 4 — Triage Breakdown\n(n={len(valid)} molecules)",
            fontsize=13, fontweight="bold",
        )
        pie_path = os.path.join(output_dir, "sa_score_pie_chart.png")
        fig2.tight_layout()
        fig2.savefig(pie_path, dpi=150)
        plt.close(fig2)
        print(f"  Pie chart saved to:   {pie_path}")

    # ------------------------------------------------------------------
    # 3. Scatter: SA score vs. heavy atom count
    # ------------------------------------------------------------------
    scatter_data = [
        (r.sa_score, r.num_heavy_atoms, r.decision)
        for r in valid
        if r.num_heavy_atoms is not None
    ]
    if scatter_data:
        fig3, ax3 = plt.subplots(figsize=(9, 6))
        for decision in ("PASS", "FLAG", "DISCARD"):
            xs = [d[0] for d in scatter_data if d[2] == decision]
            ys = [d[1] for d in scatter_data if d[2] == decision]
            if xs:
                ax3.scatter(xs, ys, c=colors[decision], label=decision,
                            s=80, alpha=0.8, edgecolors="white", linewidth=0.5)

        ax3.axvline(thresholds.pass_threshold, color="#f39c12",
                    linestyle="--", linewidth=1.2, alpha=0.7)
        ax3.axvline(thresholds.flag_threshold, color="#e74c3c",
                    linestyle="--", linewidth=1.2, alpha=0.7)
        ax3.set_xlabel("SA Score", fontsize=12)
        ax3.set_ylabel("Heavy Atom Count", fontsize=12)
        ax3.set_title("SA Score vs. Molecular Size", fontsize=13, fontweight="bold")
        ax3.legend(fontsize=10)
        ax3.grid(alpha=0.3)
        scatter_path = os.path.join(output_dir, "sa_score_complexity.png")
        fig3.tight_layout()
        fig3.savefig(scatter_path, dpi=150)
        plt.close(fig3)
        print(f"  Scatter plot saved to: {scatter_path}")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRI_FLAG Week 4: SA Score Distribution Analysis"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="CSV file with SMILES (columns: molecule_id, smiles[, name])")
    parser.add_argument("--smiles-col", type=str, default="smiles",
                        help="Column name for SMILES in input CSV (default: smiles)")
    parser.add_argument("--output-dir", type=str, default="analysis/output",
                        help="Directory for plots and CSV (default: analysis/output)")
    parser.add_argument("--context", type=str, default="default",
                        choices=list(THRESHOLD_SETS.keys()),
                        help="Threshold context: default | lead_opt | nat_prod | fragment")
    parser.add_argument("--benchmarks", action="store_true",
                        help="Run and print benchmark validation report")
    args = parser.parse_args()

    thresholds = THRESHOLD_SETS[args.context]
    print(f"\nContext: {args.context}  "
          f"(PASS<{thresholds.pass_threshold}, "
          f"FLAG<={thresholds.flag_threshold}, "
          f"DISCARD>{thresholds.flag_threshold})")

    if args.benchmarks:
        print_benchmark_report()

    if args.input:
        print(f"\nLoading molecules from: {args.input}")
        molecules = load_molecules_from_csv(args.input, smiles_col=args.smiles_col)
        print(f"  Loaded {len(molecules)} molecules.")
    else:
        print("\nUsing built-in 30-molecule reference set.")
        molecules = REFERENCE_MOLECULES

    print("Computing SA scores + complexity breakdowns...")
    results = compute_batch(molecules, thresholds=thresholds)

    print_summary(results, thresholds=thresholds)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "sa_score_results.csv")
    export_csv(results, csv_path)

    print("Generating plots...")
    plot_all(results, thresholds=thresholds, output_dir=args.output_dir)

    print("\nDone. Week 4 analysis complete.\n")


if __name__ == "__main__":
    main()