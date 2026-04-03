# loop/triflag_scorer.py
"""
ACEGEN-compatible scoring interface for TRI_FLAG.

This module exposes a single public function, triflag_score(), which ACEGEN
calls during RL training to evaluate every molecule it generates. The function
accepts a list of SMILES strings and returns a list of reward floats in [0.0,
1.0], one per input, in the same order.

Interface contract (CRITICAL — violations silently kill ACEGEN training):
  - Input:  list[str]   — one SMILES string per element
  - Output: list[float] — SAME LENGTH as input, SAME ORDER
  - All output values must be float, in [0.0, 1.0]
  - Invalid or unparseable SMILES → 0.0, never raises
  - The function itself must never raise — all exceptions caught internally

Module-level configuration (set before each generation run):
  BATCH_ID            str | None  — ACEGEN batch identifier (e.g. "gen_001")
  GENERATION_NUMBER   int | None  — generation counter for oracle tracking
  SKIP_SIMILARITY     bool        — True for generation 0 (no live API calls)
  DB_PATH             str         — path to the SQLite database file
  ENABLE_DEEPPURPOSE  bool        — True (default) to include S_act in reward.
                                    False → S_act=1.0, exactly Week 10 behaviour,
                                    zero performance overhead from DeepPurpose.

Active pipeline (5 tools):
    ValidityTool → DescriptorTool → SAScoreTool → SimilarityTool* → PAINSTool
    (*SimilarityTool skipped when SKIP_SIMILARITY=True)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CRITICAL: set TRIFLAG_ENABLE_DEEPPURPOSE in os.environ BEFORE any other
# import fires.  reporting.scoring reads this env var at module import time
# to decide whether to import DeepPurpose/torch.  If it's already imported
# (cached in sys.modules) from a previous import, the env var has no effect
# — but in normal usage this module is the first importer so the order matters.
# ---------------------------------------------------------------------------
import os

_dp_env = os.environ.get("TRIFLAG_ENABLE_DEEPPURPOSE", "1")
# Normalise: anything other than "0" means enabled
if _dp_env == "0":
    os.environ["TRIFLAG_ENABLE_DEEPPURPOSE"] = "0"
else:
    os.environ["TRIFLAG_ENABLE_DEEPPURPOSE"] = "1"

# ---------------------------------------------------------------------------
# All other imports come AFTER the env var is set
# ---------------------------------------------------------------------------
import logging
import uuid
from typing import List, Optional

from agent.triage_agent import TriageAgent
from database.db import DatabaseManager
from policies.policy_engine import PolicyEngine
from reporting.rationale_builder import RationaleBuilder
from reporting.run_record import RunRecordBuilder
from tools.sa_score_tool import SAScoreTool
from tools.similarity_tool import SimilarityTool
from tools.validity_tool import ValidityTool
from policies.thresholds import DEFAULT_SA_THRESHOLDS

from tools.descriptor_tool import DescriptorTool
from tools.pains_tool import PAINSTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level configuration — set these before each generation run.
# They are intentionally module-level globals (not function parameters) so
# that ACEGEN, which controls when and how triflag_score() is called, has no
# way to pass them as arguments. The caller sets them once before launching.
# ---------------------------------------------------------------------------

BATCH_ID: Optional[str] = os.environ.get("TRIFLAG_BATCH_ID", None)
GENERATION_NUMBER: Optional[int] = int(os.environ["TRIFLAG_GENERATION_NUMBER"]) if "TRIFLAG_GENERATION_NUMBER" in os.environ else None
SKIP_SIMILARITY: bool = os.environ.get("TRIFLAG_SKIP_SIMILARITY", "0") == "1"
DB_PATH: str = os.environ.get("TRIFLAG_DB_PATH", "runs/triflag.db")

# Week 11: whether S_act (DeepPurpose binding affinity) is included in reward.
# Reads the same env var that was set above before the reporting imports.
ENABLE_DEEPPURPOSE: bool = os.environ.get("TRIFLAG_ENABLE_DEEPPURPOSE", "1") != "0"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_agent(skip_similarity: bool = False) -> TriageAgent:
    """
    Construct a fresh TriageAgent with the current Week 10 tool pipeline.

    Full pipeline (5 tools):
        ValidityTool → DescriptorTool → SAScoreTool → SimilarityTool* → PAINSTool
        (*SimilarityTool omitted when skip_similarity=True)

    Pipeline order rationale:
        - DescriptorTool runs before SAScoreTool so scaffold_smiles is populated
          early and Lipinski checks are available to the policy engine.
        - PAINSTool runs last — it is advisory only and must not block the
          similarity check from running.
        - SimilarityTool is placed before PAINSTool so IP risk is assessed on
          the full structure before PAINS advisory flags are added.

    A new agent is constructed per triflag_score() call so that module-level
    state does not leak between ACEGEN batches.
    """
    tools = [
        ValidityTool(),
        DescriptorTool(),
        SAScoreTool(thresholds=DEFAULT_SA_THRESHOLDS),
    ]
    if not skip_similarity:
        tools.append(SimilarityTool(flag_threshold=0.90))
    tools.append(PAINSTool())

    policy_engine = PolicyEngine(sa_thresholds=DEFAULT_SA_THRESHOLDS)

    return TriageAgent(
        tools=tools,
        policy_engine=policy_engine,
        logger=logging.getLogger("triflag.agent"),
    )


def _score_one(
    smiles: str,
    agent: TriageAgent,
    rationale_builder: RationaleBuilder,
    record_builder: RunRecordBuilder,
    db: DatabaseManager,
) -> float:
    """
    Run the full TRI_FLAG pipeline on a single SMILES string.

    Returns a float in [0.0, 1.0]. All exceptions are caught; invalid input
    returns 0.0. This function is deliberately not exposed as a public API —
    it exists only to keep triflag_score() readable.
    """
    molecule_id = f"acegen_{uuid.uuid4().hex[:12]}"

    state = agent.run(molecule_id=molecule_id, raw_input=smiles)
    explanation = rationale_builder.build(state)
    record = record_builder.build(state, explanation)

    record.batch_id = BATCH_ID
    record.generation_number = GENERATION_NUMBER
    record.entry_point = "acegen"

    record.save(DB_PATH)

    reward = record.reward
    if reward is None:
        return 0.0

    return float(max(0.0, min(1.0, reward)))


# ---------------------------------------------------------------------------
# Public interface — the single function ACEGEN calls
# ---------------------------------------------------------------------------

def triflag_score(smiles: List[str]) -> List[float]:
    """
    ACEGEN-compatible scoring function for TRI_FLAG.

    Accepts a list of SMILES strings generated by ACEGEN during RL training
    and returns a list of reward floats, one per input molecule, in the same
    order. Every scored molecule is written to SQLite so the full generation
    history is available in the oracle dashboard and via MCP.

    Args:
        smiles: List of SMILES strings to score. May be empty.

    Returns:
        List of floats in [0.0, 1.0], same length and order as input.
        Invalid SMILES → 0.0. Never raises.

    Configuration (set at module level before calling):
        BATCH_ID, GENERATION_NUMBER, SKIP_SIMILARITY, DB_PATH,
        ENABLE_DEEPPURPOSE

    Example:
        >>> import loop.triflag_scorer as scorer
        >>> scorer.BATCH_ID = "gen_001"
        >>> scorer.GENERATION_NUMBER = 1
        >>> scorer.SKIP_SIMILARITY = False
        >>> scorer.triflag_score(["CCO", "CC(=O)Oc1ccccc1C(=O)O"])
        [0.0, 0.0]   # ethanol and aspirin both score 0 — novelty collapse
    """
    scores: List[float] = [0.0] * len(smiles)

    if not smiles:
        return scores

    try:
        agent = _build_agent(skip_similarity=SKIP_SIMILARITY)
        rationale_builder = RationaleBuilder()
        record_builder = RunRecordBuilder()
        db = DatabaseManager(DB_PATH)
    except Exception as exc:
        logger.error(
            "triflag_score: failed to initialise pipeline: %s", exc, exc_info=True
        )
        return scores

    for i, smi in enumerate(smiles):
        try:
            scores[i] = _score_one(
                smiles=smi,
                agent=agent,
                rationale_builder=rationale_builder,
                record_builder=record_builder,
                db=db,
            )
        except Exception as exc:
            logger.warning(
                "triflag_score: unhandled exception for SMILES %r at index %d: %s",
                smi,
                i,
                exc,
                exc_info=True,
            )
            scores[i] = 0.0

    logger.info(
        "triflag_score: batch complete — %d molecules, "
        "mean=%.4f, batch_id=%s, generation=%s, deeppurpose=%s",
        len(scores),
        sum(scores) / len(scores) if scores else 0.0,
        BATCH_ID,
        GENERATION_NUMBER,
        ENABLE_DEEPPURPOSE,
    )

    return scores