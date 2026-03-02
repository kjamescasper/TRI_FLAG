"""
mcp_server.py — TRI_FLAG MCP Server

Wraps the TRI_FLAG triage pipeline as an MCP tool so Claude Desktop
can call it directly from chat.

Place this file in triage_agent/ (same level as main.py and streamlit_app.py).

Run command (Claude Desktop handles this via claude_desktop_config.json):
    conda run -n triflag python mcp_server.py

Tool exposed:
    triage_molecule(smiles, molecule_id?, skip_similarity?) -> full rationale text
"""

import logging
import os
import sys

# ---------------------------------------------------------------------------
# Path setup — must be before any triage_agent imports
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# MCP import — FastMCP is the modern high-level API
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
from tools.similarity_tool import SimilarityTool

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


@mcp.tool()
def triage_molecule(
    smiles: str,
    molecule_id: str = "mcp_mol",
    skip_similarity: bool = False,
) -> str:
    """
    Run a molecule through the TRI_FLAG triage pipeline.

    Evaluates chemical validity, synthetic accessibility (SA score),
    and IP similarity against ChEMBL and PubChem.
    Returns PASS, FLAG, or DISCARD with a full plain-English rationale.

    Args:
        smiles: SMILES string representing the molecule (e.g. 'CCO' for ethanol)
        molecule_id: Optional identifier for tracking (default: mcp_mol)
        skip_similarity: Skip IP similarity check for faster offline use (default: False)
    """
    # Build tool list
    tools = [ValidityTool(), SAScoreTool(thresholds=DEFAULT_SA_THRESHOLDS)]
    if not skip_similarity:
        tools.append(SimilarityTool(flag_threshold=0.85))

    # Run pipeline
    agent = TriageAgent(
        tools=tools,
        policy_engine=PolicyEngine(sa_thresholds=DEFAULT_SA_THRESHOLDS),
        logger=logging.getLogger("agent.triage_agent"),
    )

    state = agent.run(molecule_id=molecule_id, raw_input=smiles)
    explanation = RationaleBuilder().build(state)
    record = RunRecordBuilder().build(state, explanation)

    # Persist to same JSONL used by CLI and Streamlit
    os.makedirs(os.path.dirname(_RUNS_FILE), exist_ok=True)
    save(record, _RUNS_FILE)

    return format_text(explanation)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")