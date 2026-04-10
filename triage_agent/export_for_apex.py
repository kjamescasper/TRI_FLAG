"""
export_for_apex.py

Exports TRI_FLAG SQLite data to clean CSVs ready for Oracle APEX upload.

Three files are produced:

  exports/triflag_molecules.csv        — full triage_runs table (all decisions)
  exports/triflag_generations.csv      — per-generation summary statistics
  exports/triflag_top_candidates.csv   — PASS only, reward >= 0.40, top candidates

Usage (from triage_agent/ directory):
    python export_for_apex.py

Optional flags:
    --db          path to triflag.db  (default: runs/triflag.db)
    --out         output directory    (default: exports/)
    --min-reward  minimum reward for top candidates (default: 0.40)

APEX compatibility notes:
  - row_num integer surrogate key added (APEX link/form operations need numeric PK)
  - Timestamps formatted as YYYY-MM-DD HH:MM:SS (APEX DATE format, no T or timezone)
  - Orphan pre-pipeline test rows filtered (novel_w9, novel_001 — no generation_number)
  - Boolean columns (is_valid, pains_alert) exported as 0/1 integers
  - NULL values exported as empty cells (APEX treats blank as NULL)
  - Column headers are lowercase with underscores (Oracle-friendly)
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults — resolved relative to current working directory so the script
# works without flags when run from triage_agent/:
#   python export_for_apex.py
# ---------------------------------------------------------------------------
_HERE        = Path.cwd()
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


def _write_csv(path: Path, rows: list, columns: list[str]) -> int:
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

def export_molecules(conn: sqlite3.Connection, out_dir: Path) -> tuple:
    """
    All triage_runs rows joined with molecules.
    Includes every decision (PASS, FLAG, DISCARD).

    APEX fixes:
      - row_num: integer surrogate key for APEX link/form operations
      - triaged_at: YYYY-MM-DD HH:MM:SS format (no T or timezone offset)
      - Filters out orphan pre-pipeline rows (novel_w9, novel_001) that have
        no generation_number and would appear as unlinked records in APEX
    """
    cursor = conn.execute("""
        SELECT
            ROW_NUMBER() OVER (
                ORDER BY tr.generation_number ASC, tr.reward DESC
            )                                                           AS row_num,
            tr.run_id,
            tr.molecule_id,
            m.canonical_smiles,
            tr.batch_id,
            tr.generation_number,
            REPLACE(SUBSTR(tr.triaged_at, 1, 19), 'T', ' ')            AS triaged_at,
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
        WHERE tr.generation_number IS NOT NULL
          AND m.canonical_smiles IS NOT NULL
        ORDER BY tr.generation_number ASC, tr.reward DESC NULLS LAST
    """)

    columns = [
        "row_num", "run_id", "molecule_id", "canonical_smiles", "batch_id",
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

def export_generations(conn: sqlite3.Connection, out_dir: Path) -> tuple:
    """
    One row per batch_id with aggregate statistics.
    Mirrors the get_all_generations_summary MCP tool output.
    batch_id is the natural primary key for this table in APEX.
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
          AND generation_number IS NOT NULL
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
) -> tuple:
    """
    PASS decisions only, reward >= min_reward, generation_number not null.
    Includes physicochemical properties for APEX table views.
    Sorted by reward DESC so highest candidates appear first.
    triaged_at formatted as YYYY-MM-DD HH:MM:SS for APEX DATE type.
    """
    cursor = conn.execute("""
        SELECT
            ROW_NUMBER() OVER (ORDER BY tr.reward DESC)                 AS row_num,
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
            REPLACE(SUBSTR(tr.triaged_at, 1, 19), 'T', ' ')            AS triaged_at
        FROM triage_runs tr
        JOIN molecules m USING (molecule_id)
        WHERE tr.final_decision = 'PASS'
          AND tr.reward >= ?
          AND tr.generation_number IS NOT NULL
          AND m.canonical_smiles IS NOT NULL
        ORDER BY tr.reward DESC
    """, (min_reward,))

    columns = [
        "row_num", "molecule_id", "canonical_smiles", "batch_id",
        "generation_number", "final_decision", "reward",
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
        help=f"Path to triflag.db (default: runs/triflag.db)",
    )
    parser.add_argument(
        "--out", type=Path, default=_DEFAULT_OUT,
        help=f"Output directory (default: exports/)",
    )
    parser.add_argument(
        "--min-reward", type=float, default=0.40,
        help="Minimum reward for top_candidates export (default: 0.40)",
    )
    args = parser.parse_args()

    print(f"\n[apex_export] TRI_FLAG -> Oracle APEX Export")
    print(f"  Database : {args.db}")
    print(f"  Output   : {args.out}")
    print(f"  Min reward (top candidates): {args.min_reward}\n")

    conn = _connect(args.db)

    # --- molecules ---
    path, n = export_molecules(conn, args.out)
    print(f"  [1/3] triflag_molecules.csv      {n:>7,} rows  ->  {path}")

    # --- generations ---
    path, n = export_generations(conn, args.out)
    print(f"  [2/3] triflag_generations.csv    {n:>7,} rows  ->  {path}")

    # --- top candidates ---
    path, n = export_top_candidates(conn, args.out, args.min_reward)
    print(f"  [3/3] triflag_top_candidates.csv {n:>7,} rows  ->  {path}")

    conn.close()

    # --- export manifest ---
    manifest_path = args.out / "export_manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("TRI_FLAG Oracle APEX Export\n")
        f.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
        f.write(f"Database:  {args.db}\n")
        f.write(f"Min reward (top candidates): {args.min_reward}\n\n")
        f.write("Files:\n")
        f.write("  triflag_molecules.csv      - all triage_runs, generation rows only\n")
        f.write("  triflag_generations.csv    - per-generation aggregate stats\n")
        f.write(f"  triflag_top_candidates.csv - PASS only, reward >= {args.min_reward}\n\n")
        f.write("APEX upload order:\n")
        f.write("  1. triflag_generations.csv  (summary table - load first)\n")
        f.write("  2. triflag_top_candidates.csv\n")
        f.write("  3. triflag_molecules.csv    (largest - load last)\n\n")
        f.write("Primary keys for APEX:\n")
        f.write("  triflag_molecules.csv      - row_num (integer) or run_id (text UUID)\n")
        f.write("  triflag_generations.csv    - batch_id (text)\n")
        f.write("  triflag_top_candidates.csv - row_num (integer) or molecule_id (text)\n")

    print(f"\n  Manifest  ->  {manifest_path}")
    print(f"\n[apex_export] Done.\n")


if __name__ == "__main__":
    main()