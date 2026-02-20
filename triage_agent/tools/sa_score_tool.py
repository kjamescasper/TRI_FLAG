"""
tools/sa_score_tool.py

Week 4: Synthetic Accessibility (SA) Scoring Tool for TRI_FLAG.

This tool is the agent-infrastructure layer. Chemistry logic lives in
chemistry/sa_score.py — this module only handles state I/O, error
formatting, and logging.

Pipeline contract:
    - Depends on ValidityTool having run first (reads canonical SMILES
      from state.tool_results['ValidityTool']).
    - Does NOT call state.terminate() itself. It stores its result and
      returns. PolicyEngine decides whether to DISCARD/FLAG/PASS based
      on the result. Early termination is wired in triage_agent.py.

Result format stored in state.tool_results['SAScoreTool']:
    See SAScoreResult dataclass below.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rdkit import Chem

from molecule import Molecule
from tools.base_tool import Tool
from agent.agent_state import AgentState
from chemistry.sa_score import (
    calculate_sa_score,
    get_complexity_breakdown,
)
from policies.thresholds import (
    DEFAULT_SA_THRESHOLDS,
    SAScoreThresholds,
)


# ---------------------------------------------------------------------------
# Structured result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SAScoreResult:
    """
    Structured output from the SA score tool.

    This dataclass is the canonical result format. The tool also returns
    it as a plain dict (via .to_dict()) for compatibility with AgentState
    which stores raw dicts in tool_results.

    Attributes:
        tool_name:               Always "SAScoreTool".
        molecule_id:             Identifier of the molecule being scored.
        sa_score:                Float in [1, 10]. None if computation failed.
        synthesizability_category: 4-tier label: easy / moderate / difficult /
                                   very_difficult. Descriptive only.
        sa_decision:             3-tier pipeline routing: PASS / FLAG / DISCARD
                                 (or ERROR if computation failed).
        sa_description:          Human-readable description of decision + category.
        complexity_breakdown:    Dict of contributing complexity factors.
                                 See chemistry/sa_score.get_complexity_breakdown().
        warning_flags:           List of synthesis challenge strings.
                                 Duplicated from complexity_breakdown for
                                 convenient top-level access.
        error_message:           Non-None only if sa_decision == "ERROR".
        execution_time_ms:       Wall-clock time for this tool run.
    """
    tool_name: str
    molecule_id: str
    sa_score: Optional[float]
    synthesizability_category: Optional[str]  # "easy"|"moderate"|"difficult"|"very_difficult"
    sa_decision: str  # "PASS"|"FLAG"|"DISCARD"|"ERROR"
    sa_description: str
    complexity_breakdown: Optional[Dict[str, Any]]
    warning_flags: List[str]
    error_message: Optional[str]
    execution_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for storage in AgentState.tool_results."""
        return {
            "tool_name": self.tool_name,
            "molecule_id": self.molecule_id,
            "sa_score": self.sa_score,
            "synthesizability_category": self.synthesizability_category,
            "sa_decision": self.sa_decision,
            "sa_description": self.sa_description,
            "complexity_breakdown": self.complexity_breakdown,
            "warning_flags": self.warning_flags,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
        }


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------

class SAScoreTool(Tool):
    """
    Computes the Ertl-Schuffenhauer Synthetic Accessibility (SA) score.

    Pipeline position: immediately after ValidityTool.

    This tool does NOT terminate the pipeline itself — it populates
    state.tool_results['SAScoreTool'] and returns. The TriageAgent reads
    PolicyEngine's evaluation of that result and handles DISCARD/FLAG.

    Args:
        thresholds: Optional custom SAScoreThresholds. Defaults to
                    DEFAULT_SA_THRESHOLDS (pass=6, flag=7).
    """

    def __init__(self, thresholds: Optional[SAScoreThresholds] = None):
        self.name = "SAScoreTool"
        self.thresholds = thresholds or DEFAULT_SA_THRESHOLDS
        self.logger = logging.getLogger(__name__)

    def run(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute SA score calculation.

        Reads canonical SMILES from ValidityTool result if available,
        falls back to raw_input. Returns a dict suitable for storage in
        state.tool_results.
        """
        t0 = time.perf_counter()
        molecule_id = state.molecule_id

        # ------------------------------------------------------------------
        # 1. Guard: ensure ValidityTool ran and molecule is valid
        # ------------------------------------------------------------------
        validity = state.tool_results.get("ValidityTool", {})
        if validity and not validity.get("is_valid", True):
            elapsed = (time.perf_counter() - t0) * 1000
            return self._error(
                molecule_id,
                "Molecule failed validity check — SA score skipped.",
                elapsed,
            ).to_dict()

        # ------------------------------------------------------------------
        # 2. Extract SMILES
        # ------------------------------------------------------------------
        smiles = self._extract_smiles(state)
        if smiles is None:
            elapsed = (time.perf_counter() - t0) * 1000
            return self._error(
                molecule_id,
                "Could not extract SMILES from state. Ensure ValidityTool ran first "
                "or provide SMILES as raw_input.",
                elapsed,
            ).to_dict()

        # ------------------------------------------------------------------
        # 3. Build RDKit mol and compute SA score + breakdown
        # ------------------------------------------------------------------
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            elapsed = (time.perf_counter() - t0) * 1000
            return self._error(
                molecule_id,
                f"RDKit could not parse SMILES: '{smiles}'",
                elapsed,
            ).to_dict()

        try:
            sa_score = calculate_sa_score(mol)
            breakdown = get_complexity_breakdown(mol)
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return self._error(molecule_id, str(exc), elapsed).to_dict()

        elapsed = (time.perf_counter() - t0) * 1000

        # ------------------------------------------------------------------
        # 4. Classify and annotate
        # ------------------------------------------------------------------
        decision = self.thresholds.classify(sa_score)
        category = self.thresholds.categorize(sa_score)
        description = self.thresholds.describe(sa_score)
        warning_flags: List[str] = breakdown.get("warning_flags", [])

        self._log(molecule_id, sa_score, decision, category, elapsed)

        result = SAScoreResult(
            tool_name=self.name,
            molecule_id=molecule_id,
            sa_score=round(sa_score, 4),
            synthesizability_category=category,
            sa_decision=decision,
            sa_description=description,
            complexity_breakdown=breakdown,
            warning_flags=warning_flags,
            error_message=None,
            execution_time_ms=round(elapsed, 3),
        )
        return result.to_dict()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_smiles(self, state: AgentState) -> Optional[str]:
        """
        Pull the best available SMILES from state.

        Priority:
          1. Canonical SMILES stored by ValidityTool (preferred — already sanitized)
          2. raw_input if it's a non-empty string
          3. raw_input.smiles if it's a Molecule object
          4. raw_input['smiles'] if it's a dict
        """
        canonical = state.tool_results.get("ValidityTool", {}).get("smiles_canonical")
        if canonical:
            return canonical

        raw = state.raw_input
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        if isinstance(raw, Molecule):
            return raw.smiles
        if isinstance(raw, dict):
            return raw.get("smiles")
        return None

    def _log(
        self,
        molecule_id: str,
        sa_score: float,
        decision: str,
        category: str,
        elapsed_ms: float,
    ) -> None:
        msg = (
            "[%s] SAScoreTool: SA=%.2f category=%s decision=%s (%.1f ms)"
        )
        if decision == "PASS":
            self.logger.info(msg, molecule_id, sa_score, category, decision, elapsed_ms)
        else:
            self.logger.warning(msg, molecule_id, sa_score, category, decision, elapsed_ms)

    def _error(
        self, molecule_id: str, error_message: str, elapsed_ms: float
    ) -> SAScoreResult:
        self.logger.error("[%s] SAScoreTool ERROR: %s", molecule_id, error_message)
        return SAScoreResult(
            tool_name=self.name,
            molecule_id=molecule_id,
            sa_score=None,
            synthesizability_category=None,
            sa_decision="ERROR",
            sa_description=f"SA score could not be computed: {error_message}",
            complexity_breakdown=None,
            warning_flags=[],
            error_message=error_message,
            execution_time_ms=round(elapsed_ms, 3),
        )