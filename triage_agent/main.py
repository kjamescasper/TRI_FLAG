"""
main.py

TRI_FLAG — Command-Line Entry Point

Week 3-5: Smoke-test entry point (single hardcoded molecule).
Week 6:   Full CLI with argparse. Accepts single SMILES or batch CSV.
          Builds flat rationale explanation and saves run records to JSONL.
Week 8:   Added --batch-id and --generation-number CLI arguments.
          save_record() replaced with record.save() (SQLite via DatabaseManager).
          batch_id, generation_number, entry_point stamped onto record before save.

Usage:
    # Single molecule
    python main.py --smiles "CCO" --id ethanol_001

    # Single molecule with ACEGEN batch context
    python main.py --smiles "CCO" --id ethanol_001 --batch-id gen_001 --generation-number 1

    # Batch from CSV (columns: molecule_id, smiles[, name])
    python main.py --input molecules.csv

    # Batch with generation tracking
    python main.py --input molecules.csv --batch-id gen_002 --generation-number 2

    # Skip similarity search (fast offline mode)
    python main.py --smiles "CCO" --no-similarity

    # Suppress log noise, print only the report
    python main.py --smiles "CCO" --quiet

    # Smoke test (Week 3-5 behaviour, no args needed)
    python main.py

Author: TRI_FLAG Research Team
Week: 8 (SQLite persistence, reward scoring, batch tracking)
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ── Core pipeline imports ────────────────────────────────────────────────────
from agent.triage_agent import TriageAgent
from policies.policy_engine import PolicyEngine
from tools.base_tool import Tool
from tools.validity_tool import ValidityTool
from tools.sa_score_tool import SAScoreTool
from tools.similarity_tool import SimilarityTool

# ── Reporting layer ──────────────────────────────────────────────────────────
from reporting.rationale_builder import RationaleBuilder, format_text
from reporting.run_record import RunRecordBuilder

# ── Week 8: SQLite persistence ───────────────────────────────────────────────
from database.db import DatabaseManager

logger = logging.getLogger(__name__)

# Default SQLite database path (Week 8 — replaces DEFAULT_OUTPUT jsonl path)
DEFAULT_DB_PATH = Path("runs") / "triflag.db"


# ============================================================================
# Initialisation helpers
# ============================================================================

def initialize_tools(*, use_similarity: bool = True) -> List[Tool]:
    """
    Initialise the tool registry in required execution order.

    Args:
        use_similarity: If False, SimilarityTool is omitted. Useful for fast
                        offline runs where IP screening is not needed.

    Returns:
        Ordered list of tools.
    """
    tools: List[Tool] = [
        ValidityTool(),
        SAScoreTool(),
    ]
    if use_similarity:
        tools.append(SimilarityTool())
    else:
        logger.info("SimilarityTool disabled (--no-similarity flag set).")

    logger.info(
        "Tools initialised (%d): %s",
        len(tools), [t.name for t in tools],
    )
    return tools


def initialize_policy_engine() -> PolicyEngine:
    """Initialise PolicyEngine with default thresholds."""
    return PolicyEngine()


def initialize_agent(tools: List[Tool], policy_engine: PolicyEngine) -> TriageAgent:
    """Wire tools and policy engine into a TriageAgent."""
    agent_logger = logging.getLogger("agent.triage_agent")
    return TriageAgent(tools=tools, policy_engine=policy_engine, logger=agent_logger)


# ============================================================================
# Single-molecule triage
# ============================================================================

def triage_molecule(
    agent: TriageAgent,
    smiles: str,
    molecule_id: str,
    db_path: Path,
    *,
    quiet: bool = False,
    batch_id: Optional[str] = None,
    generation_number: Optional[int] = None,
) -> bool:
    """
    Run one molecule through the full pipeline, print the report, save the record.

    Week 8: saves to SQLite via record.save() instead of JSONL.
    batch_id, generation_number, and entry_point are stamped onto the record
    before saving so the oracle dashboard can separate generations.

    Args:
        agent:             Initialised TriageAgent.
        smiles:            SMILES string to triage.
        molecule_id:       Identifier for this molecule.
        db_path:           SQLite database file path.
        quiet:             If True, suppress the text report (record still saved).
        batch_id:          Optional ACEGEN batch identifier (e.g. "gen_001").
        generation_number: Optional integer generation counter.

    Returns:
        True on success, False if an unrecoverable error occurred.
    """
    logger.info("Triaging molecule: %s (%s)", molecule_id, smiles)

    try:
        state = agent.run(molecule_id=molecule_id, raw_input=smiles)
    except Exception as exc:
        logger.error("Pipeline error for %s: %s", molecule_id, exc)
        return False

    # Build flat explanation
    rationale_builder = RationaleBuilder()
    explanation = rationale_builder.build(state)

    # Print report unless --quiet
    if not quiet:
        print(format_text(explanation))

    # Build run record (compute_reward() is called inside build())
    record_builder = RunRecordBuilder()
    record = record_builder.build(state, explanation)

    # Week 8: stamp batch context and entry point before saving
    record.batch_id = batch_id
    record.generation_number = generation_number
    record.entry_point = "cli"

    try:
        record.save(str(db_path))
        logger.info(
            "Run record saved: %s → %s (decision: %s, reward: %s)",
            molecule_id,
            db_path,
            record.final_decision,
            f"{record.reward:.4f}" if record.reward is not None else "None",
        )
    except Exception as exc:
        logger.error("Failed to save run record for %s: %s", molecule_id, exc)
        # Non-fatal: triage succeeded, persistence failed
        return False

    return True


# ============================================================================
# Batch processing
# ============================================================================

def load_molecules_from_csv(csv_path: Path) -> List[Tuple[str, str]]:
    """
    Load (molecule_id, smiles) pairs from a CSV file.

    Expected columns: molecule_id, smiles
    Optional column:  name (ignored for now, preserved for future use)

    Args:
        csv_path: Path to the input CSV.

    Returns:
        List of (molecule_id, smiles) tuples.

    Raises:
        SystemExit: If the file cannot be read or required columns are missing.
    """
    if not csv_path.exists():
        logger.error("Input CSV not found: %s", csv_path)
        sys.exit(1)

    molecules: List[Tuple[str, str]] = []
    try:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []

            if "smiles" not in headers:
                logger.error(
                    "CSV missing required 'smiles' column. Found: %s", headers
                )
                sys.exit(1)

            has_id = "molecule_id" in headers

            for i, row in enumerate(reader, start=1):
                smiles = row.get("smiles", "").strip()
                if not smiles:
                    logger.warning("Row %d: empty SMILES — skipping.", i)
                    continue
                mol_id = (
                    row.get("molecule_id", "").strip()
                    if has_id else f"mol_{i:04d}"
                )
                if not mol_id:
                    mol_id = f"mol_{i:04d}"
                molecules.append((mol_id, smiles))

    except (IOError, csv.Error) as exc:
        logger.error("Failed to read CSV %s: %s", csv_path, exc)
        sys.exit(1)

    logger.info("Loaded %d molecules from %s", len(molecules), csv_path)
    return molecules


def run_batch(
    agent: TriageAgent,
    molecules: List[Tuple[str, str]],
    db_path: Path,
    *,
    quiet: bool = False,
    batch_id: Optional[str] = None,
    generation_number: Optional[int] = None,
) -> None:
    """
    Triage a batch of molecules sequentially.

    Week 8: all records saved to SQLite. batch_id and generation_number are
    stamped on every record so the oracle dashboard can group them.
    Prints a summary table after all runs complete, including mean reward.

    Args:
        agent:             Initialised TriageAgent.
        molecules:         List of (molecule_id, smiles) pairs.
        db_path:           SQLite database file path.
        quiet:             If True, suppress per-molecule text reports.
        batch_id:          Optional ACEGEN batch identifier.
        generation_number: Optional integer generation counter.
    """
    total = len(molecules)
    results: List[Tuple[str, str, str, Optional[float]]] = []  # (mol_id, smiles, decision, reward)

    print(f"\nRunning batch of {total} molecules → {db_path}\n")
    if batch_id:
        print(f"  Batch ID: {batch_id}  Generation: {generation_number}\n")

    for i, (mol_id, smiles) in enumerate(molecules, start=1):
        print(f"[{i:>3}/{total}] {mol_id}  ({smiles[:40]}{'…' if len(smiles) > 40 else ''})")

        try:
            state = agent.run(molecule_id=mol_id, raw_input=smiles)
        except Exception as exc:
            logger.error("Pipeline error for %s: %s", mol_id, exc)
            results.append((mol_id, smiles, "ERROR", None))
            continue

        rationale_builder = RationaleBuilder()
        explanation = rationale_builder.build(state)

        if not quiet:
            print(format_text(explanation))

        record_builder = RunRecordBuilder()
        record = record_builder.build(state, explanation)

        # Week 8: stamp batch context
        record.batch_id = batch_id
        record.generation_number = generation_number
        record.entry_point = "cli"

        try:
            record.save(str(db_path))
        except Exception as exc:
            logger.error("Save failed for %s: %s", mol_id, exc)

        results.append((mol_id, smiles, explanation.decision, record.reward))

    # Week 8: write batch summary row to SQLite
    if batch_id:
        _save_batch_stats(db_path, batch_id, generation_number, results)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "═" * 66)
    print("  BATCH SUMMARY")
    print("═" * 66)
    counts: dict = {"PASS": 0, "FLAG": 0, "DISCARD": 0, "ERROR": 0}
    rewards: List[float] = []
    for mol_id, smiles, decision, reward in results:
        d = decision if decision in counts else "ERROR"
        counts[d] += 1
        if reward is not None:
            rewards.append(reward)
        badge = {"PASS": "✓", "FLAG": "⚑", "DISCARD": "✗", "ERROR": "!"}.get(d, "?")
        reward_str = f"  reward={reward:.4f}" if reward is not None else ""
        print(f"  {badge}  {mol_id:<20}  {d}{reward_str}")
    print("─" * 66)
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(
        f"  Total {total}:  "
        f"PASS {counts['PASS']}  "
        f"FLAG {counts['FLAG']}  "
        f"DISCARD {counts['DISCARD']}  "
        f"ERROR {counts['ERROR']}  "
        f"mean_reward={mean_reward:.4f}"
    )
    print("═" * 66)
    print(f"\nRun records saved to: {db_path}\n")


def _save_batch_stats(
    db_path: Path,
    batch_id: str,
    generation_number: Optional[int],
    results: List[Tuple[str, str, str, Optional[float]]],
) -> None:
    """
    Write aggregate batch statistics to the batches table.

    Called automatically at the end of run_batch() when --batch-id is set.
    """
    decisions = [r[2] for r in results]
    rewards = [r[3] for r in results if r[3] is not None]
    stats = {
        "generation_number": generation_number,
        "source": "cli",
        "molecule_count": len(results),
        "pass_count": decisions.count("PASS"),
        "flag_count": decisions.count("FLAG"),
        "discard_count": decisions.count("DISCARD"),
        "mean_reward": sum(rewards) / len(rewards) if rewards else None,
    }
    try:
        db = DatabaseManager(str(db_path))
        db.save_batch(batch_id, stats)
        logger.info("Batch stats saved: %s", batch_id)
    except Exception as exc:
        logger.warning("Failed to save batch stats for %s: %s", batch_id, exc)


# ============================================================================
# Smoke test (Week 3-5 behaviour — preserved for regression testing)
# ============================================================================

def run_smoke_test(agent: TriageAgent, db_path: Path) -> None:
    """
    Original Week 3-5 smoke test: triage ethanol (CCO).

    Preserved to ensure backwards compatibility with the old `python main.py`
    invocation. Now saves to SQLite instead of JSONL.
    """
    print("\nRunning Week 3-5 smoke test: ethanol (CCO)\n")
    triage_molecule(
        agent=agent,
        smiles="CCO",
        molecule_id="smoke_test_ethanol",
        db_path=db_path,
        quiet=False,
    )
    logger.info("Smoke test complete.")


# ============================================================================
# Logging configuration
# ============================================================================

def configure_logging(quiet: bool = False) -> None:
    """Configure logging. In quiet mode, suppress INFO logs from pipeline internals."""
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triflag",
        description="TRI_FLAG — Explainable molecular triage agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --smiles "CCO" --id ethanol_001
  python main.py --smiles "CCO" --id ethanol_001 --batch-id gen_001 --generation-number 1
  python main.py --input molecules.csv --batch-id gen_002 --generation-number 2
  python main.py --smiles "CCO" --no-similarity --quiet
  python main.py                                  (smoke test)
        """,
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--smiles", "-s",
        metavar="SMILES",
        help="SMILES string of a single molecule to triage.",
    )
    input_group.add_argument(
        "--input", "-i",
        metavar="CSV",
        type=Path,
        help="CSV file for batch triage. Required columns: molecule_id, smiles.",
    )

    parser.add_argument(
        "--id",
        metavar="ID",
        default=None,
        help="Molecule identifier for --smiles mode (default: 'mol_001').",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="DB",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database file for run records (default: {DEFAULT_DB_PATH}).",
    )

    # ── Week 8: ACEGEN batch tracking ────────────────────────────────────────
    parser.add_argument(
        "--batch-id",
        metavar="ID",
        default=None,
        help=(
            "ACEGEN generation batch identifier (e.g. 'gen_001'). "
            "Stored in SQLite so the oracle dashboard can separate generations."
        ),
    )
    parser.add_argument(
        "--generation-number",
        metavar="N",
        type=int,
        default=None,
        help=(
            "Integer generation counter for this ACEGEN run (1, 2, 3, ...). "
            "Required for generation-over-generation analytics."
        ),
    )
    # ── end Week 8 ────────────────────────────────────────────────────────────

    parser.add_argument(
        "--no-similarity",
        action="store_true",
        help="Skip SimilarityTool. Faster offline runs; no IP screening.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress text reports. Run records are still saved.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(quiet=args.quiet)

    use_similarity = not args.no_similarity
    tools = initialize_tools(use_similarity=use_similarity)
    policy_engine = initialize_policy_engine()
    agent = initialize_agent(tools=tools, policy_engine=policy_engine)

    # Week 8: --output now points to the SQLite db, not a JSONL file
    db_path: Path = args.output

    # ── Route to appropriate run mode ────────────────────────────────────────
    if args.smiles:
        molecule_id = args.id or "mol_001"
        success = triage_molecule(
            agent=agent,
            smiles=args.smiles,
            molecule_id=molecule_id,
            db_path=db_path,
            quiet=args.quiet,
            batch_id=args.batch_id,
            generation_number=args.generation_number,
        )
        sys.exit(0 if success else 1)

    elif args.input:
        molecules = load_molecules_from_csv(args.input)
        run_batch(
            agent=agent,
            molecules=molecules,
            db_path=db_path,
            quiet=args.quiet,
            batch_id=args.batch_id,
            generation_number=args.generation_number,
        )
        sys.exit(0)

    else:
        # No args — run original smoke test for backwards compatibility
        run_smoke_test(agent=agent, db_path=db_path)
        sys.exit(0)


if __name__ == "__main__":
    main()