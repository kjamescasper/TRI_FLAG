"""
mcp_server.py — TRI_FLAG MCP Server

Wraps the TRI_FLAG triage pipeline and database as MCP tools so Claude Desktop
can call them directly from chat.

Place this file in triage_agent/ (same level as main.py and streamlit_app.py).

Run command (Claude Desktop handles this via claude_desktop_config.json):
    conda run -n triflag python mcp_server.py

Tools exposed:
    triage_molecule(smiles, molecule_id?, skip_similarity?) -> full rationale text
    get_generation_stats(generation_number) -> per-generation reward statistics
    get_top_candidates(n?, batch_id?, min_reward?) -> top molecules by reward
    get_all_generations_summary() -> cross-generation comparison table
    search_by_scaffold(scaffold_smiles, limit?) -> molecules sharing a scaffold
    get_decision_breakdown(batch_id?) -> PASS/FLAG/DISCARD analysis with reward stats
    get_database_summary() -> high-level overview of everything in the DB
"""

import logging
import os
import sys
import warnings

# ---------------------------------------------------------------------------
# CRITICAL: MCP stdio transport requires a completely clean stdout.
# Any character printed to stdout before or during tool registration corrupts
# the JSON-RPC stream and causes Claude Desktop to see a partial tool list.
# Redirect ALL warnings to stderr and silence logging before any imports.
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")  # suppress all Python warnings to stderr
logging.disable(logging.CRITICAL)  # silence all loggers during import

# Redirect any stray prints to stderr during import (torchrl, rdkit, etc.)
_real_stdout = sys.stdout
sys.stdout = sys.stderr  # temporarily redirect stdout → stderr during imports

# ---------------------------------------------------------------------------
# Path setup — must be before any triage_agent imports
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# MCP import
# ---------------------------------------------------------------------------
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
from agent.triage_agent import TriageAgent
from policies.policy_engine import PolicyEngine
from policies.thresholds import DEFAULT_SA_THRESHOLDS
from reporting.rationale_builder import RationaleBuilder, format_text
from reporting.run_record import RunRecordBuilder, save
from tools.validity_tool import ValidityTool
from tools.sa_score_tool import SAScoreTool
# SimilarityTool imported lazily inside triage_molecule() to prevent
# chembl_webresource_client from making a live network call at MCP startup.

try:
    from tools.descriptor_tool import DescriptorTool
    _DESCRIPTOR_TOOL_OK = True
except ImportError:
    _DESCRIPTOR_TOOL_OK = False

try:
    from tools.pains_tool import PAINSTool
    _PAINS_TOOL_OK = True
except ImportError:
    _PAINS_TOOL_OK = False

# ---------------------------------------------------------------------------
# Logging — WARNING level only, never print to stdout (breaks stdio transport)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("triflag.mcp")

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
mcp = FastMCP("triflag")

_RUNS_FILE = os.path.join(_HERE, "runs", "triage_runs.jsonl")
_DB_PATH   = os.path.join(_HERE, "runs", "triflag.db")


# ---------------------------------------------------------------------------
# Tool 1 — triage_molecule
# ---------------------------------------------------------------------------

@mcp.tool()
def triage_molecule(
    smiles: str,
    molecule_id: str = "mcp_mol",
    skip_similarity: bool = False,
) -> str:
    """
    Run a molecule through the TRI_FLAG triage pipeline.

    Evaluates chemical validity, synthetic accessibility (SA score),
    physicochemical descriptors, IP similarity against ChEMBL and SureChEMBL,
    and PAINS structural alerts.
    Returns PASS, FLAG, or DISCARD with a full plain-English rationale.

    Args:
        smiles: SMILES string representing the molecule (e.g. 'CCO' for ethanol)
        molecule_id: Optional identifier for tracking (default: mcp_mol)
        skip_similarity: Skip IP similarity check for faster offline use (default: False)
    """
    from tools.similarity_tool import SimilarityTool

    tools = [ValidityTool()]
    if _DESCRIPTOR_TOOL_OK:
        tools.append(DescriptorTool())
    tools.append(SAScoreTool(thresholds=DEFAULT_SA_THRESHOLDS))
    if not skip_similarity:
        tools.append(SimilarityTool(flag_threshold=0.90))
    if _PAINS_TOOL_OK:
        tools.append(PAINSTool())

    agent = TriageAgent(
        tools=tools,
        policy_engine=PolicyEngine(sa_thresholds=DEFAULT_SA_THRESHOLDS),
        logger=logging.getLogger("agent.triage_agent"),
    )

    state = agent.run(molecule_id=molecule_id, raw_input=smiles)
    explanation = RationaleBuilder().build(state)
    record = RunRecordBuilder().build(state, explanation)

    os.makedirs(os.path.dirname(_RUNS_FILE), exist_ok=True)
    save(record, _RUNS_FILE)

    return format_text(explanation)


# ---------------------------------------------------------------------------
# Tool 2 — get_generation_stats
# ---------------------------------------------------------------------------

@mcp.tool()
def get_generation_stats(generation_number: int) -> str:
    """
    Query SQLite for per-generation reward statistics from an ACEGEN training run.

    Returns molecule count, mean/max/min reward, and PASS/FLAG/DISCARD breakdown.
    Useful for asking 'how is generation 3 performing?' during an active training run.

    Args:
        generation_number: The ACEGEN generation number to query (e.g. 0, 1, 2)
    """
    from database.db import DatabaseManager
    try:
        db = DatabaseManager(_DB_PATH)
        stats = db.get_generation_stats(generation_number)
    except Exception as exc:
        return f"Error querying database: {exc}"

    if not stats or stats.get("total", 0) == 0:
        return f"No data found for generation {generation_number}."

    total         = stats.get("total", 0)
    mean_reward   = stats.get("mean_reward") or 0
    max_reward    = stats.get("max_reward") or 0
    min_reward    = stats.get("min_reward") or 0
    pass_count    = stats.get("pass_count", 0)
    flag_count    = stats.get("flag_count", 0)
    discard_count = stats.get("discard_count", 0)
    mean_sa       = stats.get("mean_sa_score")

    pass_rate    = pass_count    / total * 100 if total else 0
    flag_rate    = flag_count    / total * 100 if total else 0
    discard_rate = discard_count / total * 100 if total else 0

    lines = [
        f"Generation {generation_number} — {total} molecules scored",
        f"  Reward:  mean={mean_reward:.4f}  max={max_reward:.4f}  min={min_reward:.4f}",
        f"  PASS:    {pass_count} ({pass_rate:.1f}%)",
        f"  FLAG:    {flag_count} ({flag_rate:.1f}%)",
        f"  DISCARD: {discard_count} ({discard_rate:.1f}%)",
    ]
    if mean_sa is not None:
        lines.append(f"  Mean SA score: {mean_sa:.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3 — get_top_candidates
# ---------------------------------------------------------------------------

@mcp.tool()
def get_top_candidates(
    n: int = 10,
    batch_id: str = None,
    min_reward: float = 0.0,
) -> str:
    """
    Return the top N molecules by reward score from the TRI_FLAG database.

    Useful for identifying the best drug candidates generated during ACEGEN training.
    Can be filtered to a specific generation batch or by minimum reward threshold.

    Args:
        n: Number of top candidates to return (default: 10, max: 50)
        batch_id: Filter to a specific batch e.g. 'gen_001' (default: all batches)
        min_reward: Only include molecules with reward >= this value (default: 0.0)
    """
    from database.db import DatabaseManager
    try:
        n = min(n, 50)
        db = DatabaseManager(_DB_PATH)
        rows = db.get_top_n_by_reward(n, batch_id=batch_id)
    except Exception as exc:
        return f"Error querying database: {exc}"

    if not rows:
        scope = f" in batch '{batch_id}'" if batch_id else ""
        return f"No candidates found{scope}."

    filtered = [dict(r) for r in rows if (dict(r).get("reward") or 0) >= min_reward]
    if not filtered:
        return f"No candidates found with reward >= {min_reward}."

    scope = f" from batch '{batch_id}'" if batch_id else " across all runs"
    lines = [f"Top {len(filtered)} candidates{scope}:"]
    lines.append(f"{'Rank':<5} {'Molecule ID':<26} {'Decision':<10} {'Reward':<8} {'S_sa':<6} {'S_nov':<6} {'S_qed':<6} {'Batch'}")
    lines.append("-" * 90)

    for i, r in enumerate(filtered, 1):
        mol_id    = (r.get("molecule_id") or "?")[:25]
        decision  = r.get("final_decision") or "?"
        reward    = r.get("reward") or 0.0
        s_sa      = r.get("s_sa")
        s_nov     = r.get("s_nov")
        s_qed     = r.get("s_qed")
        batch     = r.get("batch_id") or "?"
        s_sa_str  = f"{s_sa:.3f}"  if s_sa  is not None else "  — "
        s_nov_str = f"{s_nov:.3f}" if s_nov is not None else "  — "
        s_qed_str = f"{s_qed:.3f}" if s_qed is not None else "  — "
        lines.append(
            f"{i:<5} {mol_id:<26} {decision:<10} {reward:<8.4f} {s_sa_str:<6} {s_nov_str:<6} {s_qed_str:<6} {batch}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4 — get_all_generations_summary
# ---------------------------------------------------------------------------

@mcp.tool()
def get_all_generations_summary() -> str:
    """
    Return a cross-generation comparison table showing how ACEGEN training
    is progressing over time.

    Shows mean reward, pass rate, and molecule count for every generation.
    Use this to see whether the generator is learning and improving.
    """
    import sqlite3
    try:
        conn = sqlite3.connect(_DB_PATH)
        rows = conn.execute("""
            SELECT
                generation_number,
                batch_id,
                COUNT(*) AS total,
                ROUND(AVG(reward), 4) AS mean_reward,
                ROUND(MAX(reward), 4) AS max_reward,
                SUM(CASE WHEN final_decision='PASS'    THEN 1 ELSE 0 END) AS pass_count,
                SUM(CASE WHEN final_decision='FLAG'    THEN 1 ELSE 0 END) AS flag_count,
                SUM(CASE WHEN final_decision='DISCARD' THEN 1 ELSE 0 END) AS discard_count,
                ROUND(AVG(sa_score), 3) AS mean_sa
            FROM triage_runs
            WHERE generation_number IS NOT NULL
            GROUP BY generation_number, batch_id
            ORDER BY generation_number
        """).fetchall()
        conn.close()
    except Exception as exc:
        return f"Error querying database: {exc}"

    if not rows:
        return "No generation data found. Run at least one ACEGEN generation first."

    lines = ["Generation-over-generation summary:"]
    lines.append(f"{'Gen':<5} {'Batch':<12} {'Total':<7} {'Mean Rew':<10} {'Max Rew':<9} {'Pass%':<7} {'Disc%':<7} {'Mean SA'}")
    lines.append("-" * 75)

    for row in rows:
        gen, batch, total, mean_r, max_r, passes, flags, discards, mean_sa = row
        pass_pct = passes / total * 100 if total else 0
        disc_pct = discards / total * 100 if total else 0
        batch_str = (batch or "?")[:11]
        sa_str = f"{mean_sa:.3f}" if mean_sa else "  —  "
        lines.append(
            f"{gen:<5} {batch_str:<12} {total:<7} {(mean_r or 0):<10} {(max_r or 0):<9} "
            f"{pass_pct:<7.1f} {disc_pct:<7.1f} {sa_str}"
        )

    if len(rows) > 1:
        first_reward = rows[0][3] or 0
        last_reward  = rows[-1][3] or 0
        delta = last_reward - first_reward
        trend = "↑" if delta > 0.001 else "↓" if delta < -0.001 else "→"
        lines.append(f"\nReward trend gen {rows[0][0]} → gen {rows[-1][0]}: {trend} {delta:+.4f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 5 — search_by_scaffold
# ---------------------------------------------------------------------------

@mcp.tool()
def search_by_scaffold(
    scaffold_smiles: str,
    limit: int = 20,
) -> str:
    """
    Find all molecules in the database that share a given Bemis-Murcko scaffold.

    Useful for identifying whether ACEGEN is converging on a particular
    chemical scaffold (mode collapse) or exploring diverse chemical space.

    Args:
        scaffold_smiles: The Bemis-Murcko scaffold SMILES to search for
        limit: Maximum number of results to return (default: 20)
    """
    import sqlite3
    try:
        conn = sqlite3.connect(_DB_PATH)
        rows = conn.execute("""
            SELECT molecule_id, final_decision, reward, batch_id,
                   generation_number, scaffold_smiles
            FROM triage_runs
            WHERE scaffold_smiles = ?
            ORDER BY reward DESC
            LIMIT ?
        """, (scaffold_smiles, limit)).fetchall()

        if not rows:
            rows = conn.execute("""
                SELECT molecule_id, final_decision, reward, batch_id,
                       generation_number, scaffold_smiles
                FROM triage_runs
                WHERE scaffold_smiles LIKE ?
                ORDER BY reward DESC
                LIMIT ?
            """, (f"%{scaffold_smiles}%", limit)).fetchall()
        conn.close()
    except Exception as exc:
        return f"Error querying database: {exc}"

    if not rows:
        return f"No molecules found with scaffold matching '{scaffold_smiles}'."

    lines = [f"{len(rows)} molecules with scaffold '{scaffold_smiles}':"]
    lines.append(f"{'Molecule ID':<26} {'Decision':<10} {'Reward':<8} {'Gen':<5} {'Batch'}")
    lines.append("-" * 70)
    for row in rows:
        mol_id, decision, reward, batch, gen, scaffold = row
        lines.append(
            f"{(mol_id or '?')[:25]:<26} {(decision or '?'):<10} "
            f"{(reward or 0):<8.4f} {(str(gen) if gen is not None else '?'):<5} {batch or '?'}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 6 — get_decision_breakdown
# ---------------------------------------------------------------------------

@mcp.tool()
def get_decision_breakdown(batch_id: str = None) -> str:
    """
    Return a detailed breakdown of PASS/FLAG/DISCARD decisions with reward
    statistics for each category.

    Helps understand why molecules are being rejected and whether the reward
    signal is well-calibrated.

    Args:
        batch_id: Filter to a specific batch e.g. 'gen_001' (default: all batches)
    """
    import sqlite3
    try:
        conn = sqlite3.connect(_DB_PATH)
        if batch_id:
            rows = conn.execute("""
                SELECT final_decision, COUNT(*) AS count,
                       ROUND(AVG(reward), 4), ROUND(MAX(reward), 4),
                       ROUND(AVG(sa_score), 3), ROUND(AVG(nn_tanimoto), 3)
                FROM triage_runs WHERE batch_id = ?
                GROUP BY final_decision ORDER BY count DESC
            """, (batch_id,)).fetchall()
            total = conn.execute(
                "SELECT COUNT(*) FROM triage_runs WHERE batch_id = ?", (batch_id,)
            ).fetchone()[0]
        else:
            rows = conn.execute("""
                SELECT final_decision, COUNT(*) AS count,
                       ROUND(AVG(reward), 4), ROUND(MAX(reward), 4),
                       ROUND(AVG(sa_score), 3), ROUND(AVG(nn_tanimoto), 3)
                FROM triage_runs
                GROUP BY final_decision ORDER BY count DESC
            """).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM triage_runs").fetchone()[0]
        conn.close()
    except Exception as exc:
        return f"Error querying database: {exc}"

    if not rows:
        return f"No data found" + (f" for batch '{batch_id}'." if batch_id else ".")

    scope = f" for batch '{batch_id}'" if batch_id else " across all runs"
    lines = [f"Decision breakdown{scope} ({total:,} total molecules):"]
    lines.append(f"\n{'Decision':<10} {'Count':<8} {'%':<7} {'Mean Rew':<10} {'Max Rew':<9} {'Mean SA':<9} {'Mean Tan'}")
    lines.append("-" * 70)

    for row in rows:
        decision, count, mean_r, max_r, mean_sa, mean_tan = row
        pct     = count / total * 100 if total else 0
        sa_str  = f"{mean_sa:.3f}"  if mean_sa  is not None else "  —  "
        tan_str = f"{mean_tan:.3f}" if mean_tan is not None else "  —  "
        lines.append(
            f"{(decision or '?'):<10} {count:<8} {pct:<7.1f} {(mean_r or 0):<10.4f} "
            f"{(max_r or 0):<9.4f} {sa_str:<9} {tan_str}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 7 — get_database_summary
# ---------------------------------------------------------------------------

@mcp.tool()
def get_database_summary() -> str:
    """
    Return a high-level overview of everything in the TRI_FLAG database.

    Shows total molecules, generation count, reward statistics, top batch IDs,
    and data completeness. The best first tool to run to understand your dataset.
    """
    import sqlite3
    try:
        conn = sqlite3.connect(_DB_PATH)

        total = conn.execute("SELECT COUNT(*) FROM triage_runs").fetchone()[0]
        if total == 0:
            conn.close()
            return "Database is empty. Run at least one triage or ACEGEN generation first."

        mean_r, max_r, min_r, mean_sa = conn.execute("""
            SELECT ROUND(AVG(reward),4), ROUND(MAX(reward),4),
                   ROUND(MIN(reward),4), ROUND(AVG(sa_score),3)
            FROM triage_runs
        """).fetchone()

        gen_count = conn.execute("""
            SELECT COUNT(DISTINCT generation_number) FROM triage_runs
            WHERE generation_number IS NOT NULL
        """).fetchone()[0]

        batch_rows = conn.execute("""
            SELECT batch_id, COUNT(*) AS c FROM triage_runs
            WHERE batch_id IS NOT NULL
            GROUP BY batch_id ORDER BY c DESC LIMIT 10
        """).fetchall()

        decision_map = {r[0]: r[1] for r in conn.execute(
            "SELECT final_decision, COUNT(*) FROM triage_runs GROUP BY final_decision"
        ).fetchall()}

        null_scaffold = conn.execute(
            "SELECT COUNT(*) FROM triage_runs WHERE scaffold_smiles IS NULL"
        ).fetchone()[0]

        conn.close()
    except Exception as exc:
        return f"Error querying database: {exc}"

    passes   = decision_map.get("PASS", 0)
    flags    = decision_map.get("FLAG", 0)
    discards = decision_map.get("DISCARD", 0)

    lines = [
        "TRI_FLAG Database Summary",
        "=" * 40,
        f"Total molecules:    {total:,}",
        f"Generations run:    {gen_count}",
        f"",
        f"Reward statistics:",
        f"  Mean:  {mean_r:.4f}",
        f"  Max:   {max_r:.4f}",
        f"  Min:   {min_r:.4f}",
    ]
    if mean_sa:
        lines.append(f"  Mean SA score: {mean_sa:.3f}")
    lines += [
        f"",
        f"Decision breakdown:",
        f"  PASS:    {passes:,} ({passes/total*100:.1f}%)",
        f"  FLAG:    {flags:,} ({flags/total*100:.1f}%)",
        f"  DISCARD: {discards:,} ({discards/total*100:.1f}%)",
        f"",
        f"Data completeness:",
        f"  Missing scaffold: {null_scaffold} rows",
    ]

    if batch_rows:
        lines.append(f"")
        lines.append(f"Batches (top {len(batch_rows)}):")
        for bid, count in batch_rows:
            lines.append(f"  {(bid or 'None'):<20} {count:>6} molecules")

    return "\n".join(lines)


@mcp.tool()
def analyze_top_candidates(
    n: int = 10,
    batch_id: str = None,
    min_reward: float = 0.0,
) -> str:
    """
    Return the top N molecules by reward with full scientific interpretation.

    Unlike get_top_candidates which returns a raw table, this tool explains
    what the scores mean, flags anything noteworthy about each molecule
    (Lipinski violations, PAINS alerts, IP risk, scaffold novelty), and gives
    a plain-English assessment of each candidate's scientific value.

    Useful for asking 'which of my top molecules are actually worth looking at
    and why?' during or after an ACEGEN training run.

    Args:
        n: Number of top candidates to analyze (default: 10, max: 20)
        batch_id: Filter to a specific batch e.g. 'gen_001' (default: all)
        min_reward: Only include molecules with reward >= this value (default: 0.0)
    """
    import sqlite3

    n = min(n, 20)

    try:
        from database.db import DatabaseManager
        db = DatabaseManager(_DB_PATH)
        rows = db.get_top_n_by_reward(n, batch_id=batch_id)
    except Exception as exc:
        return f"Error querying database: {exc}"

    if not rows:
        scope = f" in batch '{batch_id}'" if batch_id else ""
        return f"No candidates found{scope}."

    filtered = [dict(r) for r in rows if (dict(r).get("reward") or 0) >= min_reward]
    if not filtered:
        return f"No candidates found with reward >= {min_reward}."

    # Pull extra descriptor/flag data from SQLite for each molecule
    try:
        conn = sqlite3.connect(_DB_PATH)
        mol_ids = [r.get("molecule_id") for r in filtered]
        placeholders = ",".join("?" * len(mol_ids))
        extra_rows = conn.execute(f"""
            SELECT molecule_id, mol_weight, logp, tpsa, hbd, hba,
                   rotatable_bonds, scaffold_smiles, pains_alert,
                   nn_tanimoto, nn_source, nn_id, similarity_decision,
                   sa_score, sa_category, generation_number, batch_id
            FROM triage_runs
            WHERE molecule_id IN ({placeholders})
        """, mol_ids).fetchall()
        conn.close()
        extra = {row[0]: row for row in extra_rows}
    except Exception:
        extra = {}

    scope = f" from batch '{batch_id}'" if batch_id else " across all runs"
    lines = [
        f"Top {len(filtered)} Candidate Analysis{scope}",
        "=" * 60,
        "",
    ]

    for i, r in enumerate(filtered, 1):
        mol_id   = r.get("molecule_id") or "?"
        reward   = r.get("reward") or 0.0
        s_sa     = r.get("s_sa")
        s_nov    = r.get("s_nov")
        s_qed    = r.get("s_qed")
        decision = r.get("final_decision") or "?"
        batch    = r.get("batch_id") or "?"
        gen      = r.get("generation_number")

        ex = extra.get(mol_id)
        mw        = ex[1]  if ex else None
        logp      = ex[2]  if ex else None
        tpsa      = ex[3]  if ex else None
        hbd       = ex[4]  if ex else None
        hba       = ex[5]  if ex else None
        rotb      = ex[6]  if ex else None
        scaffold  = ex[7]  if ex else None
        pains     = ex[8]  if ex else None
        tanimoto  = ex[9]  if ex else None
        nn_source = ex[10] if ex else None
        nn_id     = ex[11] if ex else None
        sa_score  = ex[13] if ex else None
        sa_cat    = ex[14] if ex else None

        lines.append(f"#{i} — {mol_id}")
        lines.append(f"   Batch: {batch}  |  Generation: {gen if gen is not None else '—'}")
        lines.append(f"   Decision: {decision}  |  Reward: {reward:.4f}")
        lines.append("")

        # Score breakdown with plain-English interpretation
        lines.append("   Score Breakdown:")
        if s_sa is not None:
            sa_interp = (
                "excellent — very easy to synthesize" if s_sa > 0.85 else
                "good — straightforward synthesis" if s_sa > 0.65 else
                "moderate — challenging but feasible" if s_sa > 0.40 else
                "poor — difficult synthesis"
            )
            lines.append(f"     S_sa  = {s_sa:.3f}  ({sa_interp})")
        if s_nov is not None:
            nov_interp = (
                "fully novel — no close known analogues" if s_nov > 0.95 else
                "high novelty" if s_nov > 0.70 else
                "moderate novelty — some similarity to known compounds" if s_nov > 0.30 else
                "low novelty — similar to known drugs"
            )
            lines.append(f"     S_nov = {s_nov:.3f}  ({nov_interp})")
        if s_qed is not None:
            qed_interp = (
                "excellent drug-likeness" if s_qed > 0.70 else
                "good drug-likeness" if s_qed > 0.50 else
                "moderate drug-likeness" if s_qed > 0.30 else
                "poor drug-likeness"
            )
            lines.append(f"     S_qed = {s_qed:.3f}  ({qed_interp})")
        lines.append("")

        # Physicochemical properties with Lipinski assessment
        if any(v is not None for v in [mw, logp, tpsa, hbd, hba]):
            lines.append("   Physicochemical Properties:")
            if mw   is not None: lines.append(f"     MW:   {mw:.1f} Da  {'⚠ Ro5 violation (>500)' if mw > 500 else '✓ Ro5 compliant'}")
            if logp is not None: lines.append(f"     logP: {logp:.2f}    {'⚠ Ro5 violation (>5)' if logp > 5 else '✓ Ro5 compliant'}")
            if tpsa is not None:
                cns = " — good CNS penetration" if tpsa < 90 else " — limited CNS penetration" if tpsa < 140 else " — poor CNS penetration"
                lines.append(f"     TPSA: {tpsa:.1f} Å²{cns}")
            if hbd  is not None: lines.append(f"     HBD:  {hbd}       {'⚠ Ro5 violation (>5)' if hbd > 5 else '✓ Ro5 compliant'}")
            if hba  is not None: lines.append(f"     HBA:  {hba}       {'⚠ Ro5 violation (>10)' if hba > 10 else '✓ Ro5 compliant'}")
            if rotb is not None: lines.append(f"     RotB: {rotb}")
            lines.append("")

        # IP / similarity flags
        if tanimoto is not None and tanimoto > 0:
            lines.append("   IP / Similarity:")
            lines.append(f"     Nearest neighbor: {nn_id or '?'} ({nn_source or '?'})")
            lines.append(f"     Tanimoto: {tanimoto:.3f}  " + (
                "⚠ HIGH similarity — strong IP risk, review before advancing" if tanimoto >= 0.90 else
                "moderate similarity — some IP overlap, conduct freedom-to-operate search" if tanimoto >= 0.70 else
                "low similarity — likely novel territory"
            ))
            lines.append("")

        # PAINS
        if pains is not None:
            if pains:
                lines.append("   ⚠ PAINS ALERT — structural alerts detected")
                lines.append("     Validate activity with orthogonal assay before advancing.")
                lines.append("")
            else:
                lines.append("   ✓ No PAINS alerts")
                lines.append("")

        # Scaffold
        if scaffold:
            lines.append(f"   Scaffold: {scaffold}")
            lines.append("")

        # Overall assessment
        flags = []
        if s_nov is not None and s_nov < 0.30:
            flags.append("low novelty limits scientific value")
        if s_sa is not None and s_sa < 0.40:
            flags.append("difficult synthesis may block wet-lab follow-up")
        if tanimoto is not None and tanimoto >= 0.90:
            flags.append("IP similarity requires legal review")
        if pains:
            flags.append("PAINS alert requires orthogonal validation")
        if mw is not None and mw > 500:
            flags.append("MW violates Lipinski Ro5")
        if tpsa is not None and tpsa > 90:
            flags.append("TPSA may limit CNS penetration (relevant for BACE1)")

        if flags:
            lines.append(f"   ⚠ Flags: {' | '.join(flags)}")
        else:
            lines.append("   ✓ No flags — strong candidate across all dimensions")

        lines.append("")
        lines.append("-" * 60)
        lines.append("")

    # Summary statistics across the analyzed set
    rewards = [r.get("reward") or 0 for r in filtered]
    nov_vals = [r.get("s_nov") for r in filtered if r.get("s_nov") is not None]
    qed_vals = [r.get("s_qed") for r in filtered if r.get("s_qed") is not None]

    lines.append("Summary Across Analyzed Candidates:")
    lines.append(f"  Mean reward:  {sum(rewards)/len(rewards):.4f}")
    lines.append(f"  Reward range: {min(rewards):.4f} – {max(rewards):.4f}")
    if nov_vals:
        lines.append(f"  Mean S_nov:   {sum(nov_vals)/len(nov_vals):.3f}")
    if qed_vals:
        lines.append(f"  Mean S_qed:   {sum(qed_vals)/len(qed_vals):.3f}")

    return "\n".join(lines)
if __name__ == "__main__":
    # Restore real stdout now that all imports are done.
    # From this point forward stdout belongs exclusively to the MCP JSON-RPC transport.
    sys.stdout = _real_stdout
    logging.disable(logging.NOTSET)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    mcp.run(transport="stdio")