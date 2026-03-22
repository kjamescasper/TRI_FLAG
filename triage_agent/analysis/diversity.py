# analysis/diversity.py
"""
Scaffold diversity analysis — Week 9

Computes Bemis-Murcko scaffold diversity metrics for a batch of molecules.
Detects mode collapse: the primary failure mode in molecular RL where the
generator fixates on one high-scoring scaffold and stops exploring chemical space.

Usage:
    from analysis.diversity import compute_diversity
    report = compute_diversity(scaffold_smiles_list)
    if report.convergence_warning:
        print(f"Mode collapse warning: {report.top_scaffold_frequency:.0%} same scaffold")

Can be called standalone with a list of scaffold SMILES, or via oracle_dashboard.py
which queries SQLite and passes scaffold_smiles values automatically.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from collections import Counter


@dataclass
class DiversityReport:
    """
    Scaffold diversity metrics for a single batch.

    Attributes:
        unique_scaffolds:       Number of distinct Bemis-Murcko scaffolds.
        total_molecules:        Total molecules in the batch (including invalid/NULL scaffold).
        diversity_ratio:        unique_scaffolds / total_with_scaffold. 1.0 = fully diverse.
        top_scaffold:           SMILES of the most common scaffold (None if no scaffolds).
        top_scaffold_count:     Raw count of the most common scaffold.
        top_scaffold_frequency: Fraction of batch sharing the most common scaffold.
        convergence_warning:    True if top_scaffold_frequency > 0.30 (mode collapse risk).
        null_scaffold_count:    Molecules with NULL scaffold_smiles (Week 9 not yet populated).
    """
    unique_scaffolds: int
    total_molecules: int
    diversity_ratio: float
    top_scaffold: Optional[str]
    top_scaffold_count: int
    top_scaffold_frequency: float
    convergence_warning: bool
    null_scaffold_count: int = 0


# Threshold for mode collapse warning — tunable
CONVERGENCE_THRESHOLD = 0.30


def compute_diversity(scaffold_smiles: List[Optional[str]]) -> DiversityReport:
    """
    Compute scaffold diversity metrics from a list of scaffold SMILES.

    Args:
        scaffold_smiles: List of scaffold SMILES strings. None values (invalid
                         molecules or Week 9 backlog) are counted but excluded
                         from diversity calculations.

    Returns:
        DiversityReport with all metrics populated.
    """
    total = len(scaffold_smiles)

    # Separate valid scaffolds from nulls
    valid_scaffolds = [s for s in scaffold_smiles if s is not None and s.strip() != ""]
    null_count = total - len(valid_scaffolds)

    if not valid_scaffolds:
        return DiversityReport(
            unique_scaffolds=0,
            total_molecules=total,
            diversity_ratio=0.0,
            top_scaffold=None,
            top_scaffold_count=0,
            top_scaffold_frequency=0.0,
            convergence_warning=False,
            null_scaffold_count=null_count,
        )

    counter: Dict[str, int] = Counter(valid_scaffolds)
    unique_count = len(counter)
    diversity_ratio = unique_count / len(valid_scaffolds)

    top_scaffold, top_count = counter.most_common(1)[0]
    top_frequency = top_count / len(valid_scaffolds)
    convergence_warning = top_frequency > CONVERGENCE_THRESHOLD

    return DiversityReport(
        unique_scaffolds=unique_count,
        total_molecules=total,
        diversity_ratio=round(diversity_ratio, 4),
        top_scaffold=top_scaffold,
        top_scaffold_count=top_count,
        top_scaffold_frequency=round(top_frequency, 4),
        convergence_warning=convergence_warning,
        null_scaffold_count=null_count,
    )


def compute_diversity_from_db(db_path: str, batch_id: Optional[str] = None) -> DiversityReport:
    """
    Query SQLite and compute diversity for a batch (or all runs).

    Args:
        db_path:  Path to triflag.db.
        batch_id: Filter to a specific generation. None = all runs.

    Returns:
        DiversityReport.
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if batch_id is not None:
            rows = conn.execute(
                "SELECT scaffold_smiles FROM triage_runs WHERE batch_id = ?",
                (batch_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT scaffold_smiles FROM triage_runs"
            ).fetchall()
    finally:
        conn.close()

    scaffold_list = [row["scaffold_smiles"] for row in rows]
    return compute_diversity(scaffold_list)