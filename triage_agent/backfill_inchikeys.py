"""
backfill_inchikeys.py — Retroactively compute InChIKey and InChI for all
molecules in the TRI_FLAG database that are missing them.

Every molecule already has canonical_smiles — this script computes
InChI and InChIKey from that using RDKit and writes them back to the
molecules table. No rows are deleted or recreated.

After running this, re-export CSVs for Oracle APEX:
    python export_for_apex.py

Usage (from triage_agent/ directory):
    python backfill_inchikeys.py

Optional flags:
    --db      path to triflag.db (default: runs/triflag.db)
    --dry-run print counts but do not write anything
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# RDKit import — required
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
except ImportError:
    print("ERROR: RDKit not available. Activate the triflag conda environment first.")
    print("  conda activate triflag")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_HERE        = Path.cwd()
_DEFAULT_DB  = _HERE / "runs" / "triflag.db"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def backfill(db_path: Path, dry_run: bool = False) -> None:
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Count what needs filling
    total_molecules = conn.execute("SELECT COUNT(*) FROM molecules").fetchone()[0]
    missing = conn.execute("""
        SELECT COUNT(*) FROM molecules
        WHERE inchikey IS NULL
          AND canonical_smiles IS NOT NULL
    """).fetchone()[0]
    already_done = total_molecules - missing

    print(f"\n[backfill_inchikeys] Database: {db_path}")
    print(f"  Total molecules:    {total_molecules:,}")
    print(f"  Already have InChIKey: {already_done:,}")
    print(f"  Missing InChIKey:   {missing:,}")

    if missing == 0:
        print("\n  Nothing to do — all molecules already have InChIKeys.")
        conn.close()
        return

    if dry_run:
        print(f"\n  [dry-run] Would update {missing:,} rows. No changes written.")
        conn.close()
        return

    print(f"\n  Computing InChIKey for {missing:,} molecules...")

    rows = conn.execute("""
        SELECT molecule_id, canonical_smiles
        FROM molecules
        WHERE inchikey IS NULL
          AND canonical_smiles IS NOT NULL
    """).fetchall()

    updated  = 0
    failed   = 0
    skipped  = 0

    for row in rows:
        mol_id = row["molecule_id"]
        smiles = row["canonical_smiles"]

        if not smiles or not smiles.strip():
            skipped += 1
            continue

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                failed += 1
                continue

            inchi = MolToInchi(mol)
            if inchi is None:
                failed += 1
                continue

            inchikey = InchiToInchiKey(inchi)
            if inchikey is None:
                failed += 1
                continue

            conn.execute(
                "UPDATE molecules SET inchikey = ?, inchi = ? WHERE molecule_id = ?",
                (inchikey, inchi, mol_id),
            )
            updated += 1

        except Exception as exc:
            print(f"  WARNING: {mol_id} failed ({exc})")
            failed += 1

    conn.commit()
    conn.close()

    print(f"\n[backfill_inchikeys] Complete.")
    print(f"  Updated:  {updated:,}")
    print(f"  Failed:   {failed:,}  (invalid SMILES or RDKit error)")
    print(f"  Skipped:  {skipped:,}  (empty SMILES)")

    if failed > 0:
        print(f"\n  Note: {failed} failures are expected for pre-pipeline test")
        print(f"  records (novel_w9, novel_001) with non-standard SMILES.")

    print(f"\n  Next step: python export_for_apex.py")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backfill InChIKey and InChI for all molecules missing them.",
        epilog="Run from triage_agent/ directory.",
    )
    parser.add_argument(
        "--db", type=Path, default=_DEFAULT_DB,
        help=f"Path to triflag.db (default: runs/triflag.db)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print counts but do not write anything to the database.",
    )
    args = parser.parse_args()

    backfill(args.db, dry_run=args.dry_run)


if __name__ == "__main__":
    main()