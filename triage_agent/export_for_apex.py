"""
scripts/export_for_apex.py

Exports TRI_FLAG SQLite data to clean CSVs ready for Oracle APEX upload.

Three files are produced:

  exports/triflag_molecules.csv        — full triage_runs table (all decisions)
  exports/triflag_generations.csv      — per-generation summary statistics
  exports/triflag_top_candidates.csv   — PASS only, reward >= 0.40, top candidates

Usage (from triage_agent/ directory):
    python scripts/export_for_apex.py

Optional flags:
    --db       path to triflag.db  (default: runs/triflag.db)
    --out      output directory    (default: exports/)
    --min-reward  minimum reward for top candidates (default: 0.40)

APEX upload notes:
  - All timestamps are ISO 8601 (APEX reads these directly)
  - Boolean columns (is_valid, pains_alert) exported as 0/1 integers
  - NULL values exported as empty cells (APEX treats blank as NULL)
  - Column headers are lowercase with underscores (Oracle-friendly)
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_HERE        = Path.cwd()                    # wherever you run the script from
_DEFAULT_DB  = _HERE / "runs" / "triflag.db"
_DEFAULT_OUT = _HERE / "exports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _write_csv(path: Path, rows: list[sqlite3.Row], columns: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        count = 0
        for row in rows:
            writer.writerow([row[c] if c in row.keys() else "" for c in columns])
            count += 1
    return count


# ---------------------------------------------------------------------------
# Export 1 — full molecules table
# ---------------------------------------------------------------------------

def export_molecules(conn: sqlite3.Connection, out_dir: Path) -> Path:
    """
    All triage_runs rows joined with molecules.
    Includes every decision (PASS, FLAG, DISCARD).
    Columns ordered for readability in APEX.
    """
    cursor = conn.execute("""
        SELECT
            tr.run_id,
            tr.molecule_id,
            m.canonical_smiles,
            tr.batch_id,
            tr.generation_number,
            tr.triaged_at,
            tr.final_decision,
            tr.reward,
            tr.s_sa,
            tr.s_nov,
            tr.s_qed,
            tr.s_act,
            tr.sa_score,
            tr.sa_category,
            tr.nn_tanimoto,
            tr.nn_source,
            tr.nn_id,
            tr.similarity_decision,
            tr.is_valid,
            tr.mol_weight,
            tr.logp,
            tr.tpsa,
            tr.hbd,
            tr.hba,
            tr.rotatable_bonds,
            tr.scaffold_smiles,
            tr.pains_alert,
            tr.predicted_affinity,
            tr.target_id,
            tr.rationale
        FROM triage_runs tr
        JOIN molecules m USING (molecule_id)
        ORDER BY tr.generation_number ASC, tr.reward DESC NULLS LAST
    """)

    columns = [
        "run_id", "molecule_id", "canonical_smiles", "batch_id",
        "generation_number", "triaged_at", "final_decision", "reward",
        "s_sa", "s_nov", "s_qed", "s_act",
        "sa_score", "sa_category",
        "nn_tanimoto", "nn_source", "nn_id", "similarity_decision",
        "is_valid", "mol_weight", "logp", "tpsa", "hbd", "hba",
        "rotatable_bonds", "scaffold_smiles", "pains_alert",
        "predicted_affinity", "target_id", "rationale",
    ]

    rows = cursor.fetchall()
    out  = out_dir / "triflag_molecules.csv"
    n    = _write_csv(out, rows, columns)
    return out, n


# ---------------------------------------------------------------------------
# Export 2 — per-generation summary
# ---------------------------------------------------------------------------

def export_generations(conn: sqlite3.Connection, out_dir: Path) -> Path:
    """
    One row per batch_id with aggregate statistics.
    Mirrors the get_all_generations_summary MCP tool output.
    """
    cursor = conn.execute("""
        SELECT
            batch_id,
            generation_number,
            COUNT(*)                                                    AS total_molecules,
            SUM(CASE WHEN final_decision = 'PASS'    THEN 1 ELSE 0 END) AS pass_count,
            SUM(CASE WHEN final_decision = 'FLAG'    THEN 1 ELSE 0 END) AS flag_count,
            SUM(CASE WHEN final_decision = 'DISCARD' THEN 1 ELSE 0 END) AS discard_count,
            ROUND(100.0 * SUM(CASE WHEN final_decision = 'PASS' THEN 1 ELSE 0 END)
                  / COUNT(*), 2)                                        AS pass_pct,
            ROUND(AVG(reward),  4)                                      AS mean_reward,
            ROUND(MAX(reward),  4)                                      AS max_reward,
            ROUND(MIN(CASE WHEN reward > 0 THEN reward END), 4)         AS min_nonzero_reward,
            ROUND(AVG(sa_score), 3)                                     AS mean_sa_score,
            ROUND(AVG(nn_tanimoto), 4)                                  AS mean_tanimoto,
            ROUND(AVG(s_sa),  4)                                        AS mean_s_sa,
            ROUND(AVG(s_nov), 4)                                        AS mean_s_nov,
            ROUND(AVG(s_qed), 4)                                        AS mean_s_qed,
            ROUND(AVG(s_act), 4)                                        AS mean_s_act
        FROM triage_runs
        WHERE batch_id IS NOT NULL
        GROUP BY batch_id, generation_number
        ORDER BY generation_number ASC, batch_id ASC
    """)

    columns = [
        "batch_id", "generation_number", "total_molecules",
        "pass_count", "flag_count", "discard_count", "pass_pct",
        "mean_reward", "max_reward", "min_nonzero_reward",
        "mean_sa_score", "mean_tanimoto",
        "mean_s_sa", "mean_s_nov", "mean_s_qed", "mean_s_act",
    ]

    rows = cursor.fetchall()
    out  = out_dir / "triflag_generations.csv"
    n    = _write_csv(out, rows, columns)
    return out, n


# ---------------------------------------------------------------------------
# Export 3 — top candidates (PASS only, reward >= threshold)
# ---------------------------------------------------------------------------

def export_top_candidates(
    conn: sqlite3.Connection,
    out_dir: Path,
    min_reward: float = 0.40,
) -> Path:
    """
    PASS decisions only, reward >= min_reward.
    Includes physicochemical properties for APEX table views.
    Sorted by reward DESC so highest candidates appear first.
    """
    cursor = conn.execute("""
        SELECT
            tr.molecule_id,
            m.canonical_smiles,
            tr.batch_id,
            tr.generation_number,
            tr.final_decision,
            tr.reward,
            tr.s_sa,
            tr.s_nov,
            tr.s_qed,
            tr.s_act,
            tr.sa_score,
            tr.nn_tanimoto,
            tr.nn_source,
            tr.nn_id,
            tr.mol_weight,
            tr.logp,
            tr.tpsa,
            tr.hbd,
            tr.hba,
            tr.rotatable_bonds,
            tr.scaffold_smiles,
            tr.pains_alert,
            tr.predicted_affinity,
            tr.triaged_at
        FROM triage_runs tr
        JOIN molecules m USING (molecule_id)
        WHERE tr.final_decision = 'PASS'
          AND tr.reward >= ?
        ORDER BY tr.reward DESC
    """, (min_reward,))

    columns = [
        "molecule_id", "canonical_smiles", "batch_id", "generation_number",
        "final_decision", "reward",
        "s_sa", "s_nov", "s_qed", "s_act",
        "sa_score", "nn_tanimoto", "nn_source", "nn_id",
        "mol_weight", "logp", "tpsa", "hbd", "hba", "rotatable_bonds",
        "scaffold_smiles", "pains_alert", "predicted_affinity", "triaged_at",
    ]

    rows = cursor.fetchall()
    out  = out_dir / "triflag_top_candidates.csv"
    n    = _write_csv(out, rows, columns)
    return out, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export TRI_FLAG SQLite data to CSVs for Oracle APEX upload.",
        epilog="Run from triage_agent/ directory.",
    )
    parser.add_argument(
        "--db", type=Path, default=_DEFAULT_DB,
        help=f"Path to triflag.db (default: {_DEFAULT_DB})",
    )
    parser.add_argument(
        "--out", type=Path, default=_DEFAULT_OUT,
        help=f"Output directory (default: {_DEFAULT_OUT})",
    )
    parser.add_argument(
        "--min-reward", type=float, default=0.40,
        help="Minimum reward for top_candidates export (default: 0.40)",
    )
    args = parser.parse_args()

    print(f"\n[apex_export] TRI_FLAG → Oracle APEX Export")
    print(f"  Database : {args.db}")
    print(f"  Output   : {args.out}")
    print(f"  Min reward (top candidates): {args.min_reward}\n")

    conn = _connect(args.db)

    # --- molecules ---
    path, n = export_molecules(conn, args.out)
    print(f"  [1/3] triflag_molecules.csv      {n:>7,} rows  →  {path}")

    # --- generations ---
    path, n = export_generations(conn, args.out)
    print(f"  [2/3] triflag_generations.csv    {n:>7,} rows  →  {path}")

    # --- top candidates ---
    path, n = export_top_candidates(conn, args.out, args.min_reward)
    print(f"  [3/3] triflag_top_candidates.csv {n:>7,} rows  →  {path}")

    conn.close()

    # --- export manifest ---
    manifest_path = args.out / "export_manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(f"TRI_FLAG Oracle APEX Export\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Database:  {args.db}\n")
        f.write(f"Min reward (top candidates): {args.min_reward}\n\n")
        f.write("Files:\n")
        f.write("  triflag_molecules.csv      — all triage_runs, all decisions\n")
        f.write("  triflag_generations.csv    — per-generation aggregate stats\n")
        f.write(f"  triflag_top_candidates.csv — PASS only, reward >= {args.min_reward}\n\n")
        f.write("APEX upload order:\n")
        f.write("  1. triflag_generations.csv  (summary view — load first)\n")
        f.write("  2. triflag_top_candidates.csv\n")
        f.write("  3. triflag_molecules.csv    (largest — load last)\n")

    print(f"\n  Manifest  →  {manifest_path}")
    print(f"\n[apex_export] Done.\n")


if __name__ == "__main__":
    main()