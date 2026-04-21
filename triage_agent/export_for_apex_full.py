"""
export_for_apex_full.py — TRI_FLAG

Exports every table in triflag.db to a CSV file for Oracle APEX import.
One CSV per table, every column exactly as it exists in SQLite.
No joins, no computed columns, no filtering, no dropped columns.

Output files (written to exports/):
    apex_molecules.csv            — molecules table (7 cols)
    apex_triage_runs.csv          — triage_runs table (34 cols)
    apex_batches.csv              — batches table (9 cols)
    apex_target_predictions.csv   — target_predictions table (7 cols)
    apex_export_manifest.txt      — row counts, column lists, timestamp

Usage (from triage_agent/ directory):
    python export_for_apex.py
    python export_for_apex.py --db runs/triflag.db --out exports/

APEX compatibility notes:
    - Timestamps are reformatted from ISO 8601 (2026-04-07T19:23:11+00:00)
      to YYYY-MM-DD HH:MM:SS (Oracle DATE format, no T or timezone offset)
    - NULL values export as empty cells (APEX treats blank as NULL on import)
    - Boolean columns (is_valid, pains_alert) remain as 0/1 integers
    - Column headers are lowercase with underscores (Oracle-friendly)
    - All rows exported with no filtering
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_DB  = Path("runs/triflag.db")
_DEFAULT_OUT = Path("exports")

TABLES = [
    "molecules",
    "triage_runs",
    "batches",
    "target_predictions",
]

# Timestamp columns that need reformatting for Oracle DATE
TIMESTAMP_COLS = {"triaged_at", "created_at", "predicted_at"}


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _get_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Return column names in schema order via PRAGMA."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [r["name"] for r in rows]


def _format_timestamp(value: str | None) -> str:
    """Convert ISO 8601 to YYYY-MM-DD HH:MM:SS for Oracle DATE type."""
    if not value:
        return ""
    return str(value)[:19].replace("T", " ")


def _write_table(
    conn: sqlite3.Connection,
    table: str,
    out_dir: Path,
) -> tuple[Path, int, list[str]]:
    """
    Dump an entire table to CSV. Returns (output_path, row_count, columns).
    Reads columns directly from PRAGMA so the export always matches the
    live schema — no hardcoded column lists.
    """
    columns  = _get_columns(conn, table)
    out_path = out_dir / f"apex_{table}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = conn.execute(f"SELECT * FROM {table}").fetchall()

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            formatted = []
            for col in columns:
                val = row[col]
                if col in TIMESTAMP_COLS and val is not None:
                    val = _format_timestamp(val)
                elif val is None:
                    val = ""
                formatted.append(val)
            writer.writerow(formatted)

    return out_path, len(rows), columns


def _write_manifest(
    out_dir: Path,
    db_path: Path,
    results: list[tuple[str, Path, int, list[str]]],
) -> Path:
    manifest_path = out_dir / "apex_export_manifest.txt"
    now       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    total_rows = sum(r[2] for r in results)

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("TRI_FLAG — Oracle APEX Export Manifest\n")
        f.write("=" * 52 + "\n")
        f.write(f"Generated  : {now}\n")
        f.write(f"Database   : {db_path.resolve()}\n")
        f.write(f"Tables     : {len(results)}\n")
        f.write(f"Total rows : {total_rows:,}\n\n")

        f.write("Files\n")
        f.write("-" * 52 + "\n")
        for table, path, n, cols in results:
            f.write(f"\n  {path.name}\n")
            f.write(f"    Source table : {table}\n")
            f.write(f"    Rows         : {n:,}\n")
            f.write(f"    Columns ({len(cols):2d}) : {', '.join(cols)}\n")

        f.write("\n" + "-" * 52 + "\n")
        f.write("APEX upload order:\n")
        f.write("  1. apex_molecules.csv            (parent — load first)\n")
        f.write("  2. apex_batches.csv\n")
        f.write("  3. apex_target_predictions.csv\n")
        f.write("  4. apex_triage_runs.csv           (largest — load last)\n\n")
        f.write("Primary keys:\n")
        f.write("  apex_molecules.csv           — molecule_id (TEXT)\n")
        f.write("  apex_triage_runs.csv         — run_id (TEXT UUID)\n")
        f.write("  apex_batches.csv             — batch_id (TEXT)\n")
        f.write("  apex_target_predictions.csv  — prediction_id (TEXT UUID)\n\n")
        f.write("Notes:\n")
        f.write("  - Timestamps reformatted to YYYY-MM-DD HH:MM:SS (Oracle DATE)\n")
        f.write("  - NULL values exported as empty cells\n")
        f.write("  - Boolean columns (is_valid, pains_alert) remain as 0/1\n")
        f.write("  - No rows filtered, no columns added or dropped\n")
        f.write("  - Column order matches PRAGMA table_info (schema order)\n")

    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export TRI_FLAG SQLite tables to CSVs for Oracle APEX. "
            "One CSV per table, every column, no filtering."
        ),
        epilog="Run from the triage_agent/ directory.",
    )
    parser.add_argument(
        "--db", type=Path, default=_DEFAULT_DB,
        help="Path to triflag.db (default: runs/triflag.db)",
    )
    parser.add_argument(
        "--out", type=Path, default=_DEFAULT_OUT,
        help="Output directory (default: exports/)",
    )
    args = parser.parse_args()

    print(f"\nTRI_FLAG — Oracle APEX Export")
    print(f"  Database : {args.db}")
    print(f"  Output   : {args.out}\n")

    conn    = _connect(args.db)
    results = []

    for i, table in enumerate(TABLES, 1):
        out_path, n, cols = _write_table(conn, table, args.out)
        results.append((table, out_path, n, cols))
        print(f"  [{i}/{len(TABLES)}] {out_path.name:<42} {n:>7,} rows   {len(cols)} cols")

    conn.close()

    manifest = _write_manifest(args.out, args.db, results)
    print(f"\n  Manifest  ->  {manifest}")
    print(f"\nDone. {sum(r[2] for r in results):,} total rows exported.\n")


if __name__ == "__main__":
    main()