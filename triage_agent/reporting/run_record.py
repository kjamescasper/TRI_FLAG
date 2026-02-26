"""
reporting/run_record.py

TRI_FLAG Week 6 — Run Record & Persistence

Serialises a completed triage run (AgentState + TriageExplanation) into a
flat, fully-typed RunRecord and persists it to an append-mode JSONL file.

Each line in the JSONL file is one self-contained JSON object representing
a single molecule's complete triage run — safe to stream, query, or load
into pandas / a database later.

Design goals:
    - Zero loss: every field from every tool result is captured
    - Forward-compatible: new fields added as nullable, old records still load
    - Optional AI reviewer slot: present but null until Week 7+ wires it up
    - DB-ready schema: field names and types match the planned SQL schema exactly
    - Atomic writes: partial JSONL lines never appear on disk

Public API:
    build(state, explanation, *, ai_review=None) -> RunRecord
    save(record, path)                           -> None   (appends to JSONL)
    load_all(path)                               -> list[RunRecord]
    load_as_dicts(path)                          -> list[dict]

Week: 6
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AI Reviewer sub-record (nullable — filled in Week 7+)
# ---------------------------------------------------------------------------

@dataclass
class AIReviewRecord:
    """
    Structured output from an optional AI reviewer pass.

    Fields map directly to the planned DB column set so that adding
    the reviewer in Week 7 requires zero schema changes.

    Attributes:
        model:        Model string (e.g. "claude-sonnet-4-20250514").
        ai_decision:  Reviewer's recommended decision: PASS | FLAG | DISCARD.
        agrees:       True if ai_decision matches the rule-based decision.
        reasoning:    2–4 sentence explanation referencing specific scores.
        nuance:       Any subtlety the rule-based system may have missed, or None.
        confidence:   Reviewer's stated confidence: "high" | "medium" | "low".
        raw_response: Full JSON string returned by the model for audit.
        error:        Set if the API call failed; all other fields will be None.
    """
    model: str
    ai_decision: Optional[str] = None
    agrees: Optional[bool] = None
    reasoning: Optional[str] = None
    nuance: Optional[str] = None
    confidence: Optional[str] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIReviewRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# RunRecord — canonical output of one triage run
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """
    Complete, flat record of a single TRI_FLAG triage run.

    This is the canonical unit of persistence. Every field is either a
    primitive (str, float, bool, int) or a JSON-serialisable container
    (list, dict), so the whole object round-trips cleanly through JSON.

    Field groups:
        Run identity    — run_id, molecule_id, smiles_*, timestamps
        Validity        — is_valid, num_atoms, num_bonds
        SA Score        — sa_score, sa_decision, category, warnings ...
        Similarity      — nn_tanimoto, nn_source, nn_id, hit counts ...
        Decision        — final_decision, rule_rationale, flags_raised
        Explanation     — explanation_summary, explanation_sections
        AI review       — ai_review (AIReviewRecord | None)
        Run metadata    — thresholds_preset, execution_time_ms, early_termination
    """

    # ── Run identity ────────────────────────────────────────────────────────
    run_id: str                          # UUID4, unique per run
    molecule_id: str
    run_timestamp: str                   # ISO-8601 UTC

    # ── SMILES ──────────────────────────────────────────────────────────────
    smiles_input: Optional[str]          # raw_input as given to TriageAgent
    smiles_canonical: Optional[str]      # RDKit-normalised, from ValidityTool

    # ── Validity ────────────────────────────────────────────────────────────
    is_valid: Optional[bool]
    validity_error: Optional[str]
    num_atoms: Optional[int]
    num_bonds: Optional[int]

    # ── SA Score ────────────────────────────────────────────────────────────
    sa_score: Optional[float]
    sa_decision: Optional[str]           # PASS | FLAG | DISCARD | ERROR
    synthesizability_category: Optional[str]  # easy | moderate | difficult | very_difficult
    sa_description: Optional[str]
    sa_warning_flags: List[str]          # from complexity_breakdown
    sa_complexity_breakdown: Optional[Dict[str, Any]]
    sa_execution_time_ms: Optional[float]

    # ── Similarity / IP risk ────────────────────────────────────────────────
    nn_tanimoto: Optional[float]         # nearest-neighbor Tanimoto (0.0 if none)
    nn_source: Optional[str]             # "ChEMBL" | "PubChem" | None
    nn_id: Optional[str]
    nn_name: Optional[str]
    nn_smiles: Optional[str]
    similarity_decision: Optional[str]   # PASS | FLAG | ERROR
    similarity_flag_threshold: Optional[float]
    chembl_hit_count: Optional[int]
    pubchem_hit_count: Optional[int]
    chembl_hits: List[Dict[str, Any]]    # full hit list for provenance
    pubchem_hits: List[Dict[str, Any]]
    similarity_escalated: bool           # True if tanimoto >= 0.95
    apis_queried: List[str]
    similarity_execution_time_ms: Optional[float]
    similarity_error: Optional[str]

    # ── Final decision ───────────────────────────────────────────────────────
    final_decision: str                  # PASS | FLAG | DISCARD
    rule_rationale: str                  # from Decision.rationale (PolicyEngine)
    flags_raised: List[Dict[str, str]]   # [{reason, source}, ...]

    # ── Flat explanation (RationaleBuilder output) ───────────────────────────
    explanation_summary: str
    explanation_sections: List[Dict[str, Any]]  # serialised ExplanationSection list

    # ── AI reviewer (optional — null until wired in Week 7+) ────────────────
    ai_review: Optional[AIReviewRecord]

    # ── Run metadata ─────────────────────────────────────────────────────────
    early_termination: bool
    termination_reason: Optional[str]
    total_execution_time_ms: Optional[float]
    tools_executed: List[str]            # tool names that actually ran

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a fully JSON-serialisable dict.

        Handles nested dataclasses (AIReviewRecord) explicitly so that
        json.dumps() works without a custom encoder.
        """
        d = asdict(self)
        # asdict() recurses into nested dataclasses automatically
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunRecord":
        """
        Reconstruct a RunRecord from a dict (e.g. loaded from JSONL).

        Forward-compatible: unknown keys are silently ignored so that
        records written by newer versions of this code still load.
        """
        known_fields = {f for f in cls.__dataclass_fields__}

        # Reconstruct nested AIReviewRecord if present
        ai_review_data = data.get("ai_review")
        ai_review: Optional[AIReviewRecord] = None
        if isinstance(ai_review_data, dict):
            ai_review = AIReviewRecord.from_dict(ai_review_data)

        filtered = {k: v for k, v in data.items() if k in known_fields and k != "ai_review"}
        filtered["ai_review"] = ai_review
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class RunRecordBuilder:
    """
    Assembles a RunRecord from a completed AgentState and TriageExplanation.

    Usage:
        builder = RunRecordBuilder()
        record = builder.build(state, explanation)
        # or with AI review:
        record = builder.build(state, explanation, ai_review=ai_record)
    """

    def build(
        self,
        state: Any,
        explanation: Any,  # TriageExplanation — avoid circular import
        *,
        ai_review: Optional[AIReviewRecord] = None,
    ) -> RunRecord:
        """
        Build a RunRecord from a finalized AgentState and TriageExplanation.

        Args:
            state:       AgentState returned by TriageAgent.run().
            explanation: TriageExplanation from RationaleBuilder.build(state).
            ai_review:   Optional AIReviewRecord. Pass None (default) until
                         the AI reviewer is wired in Week 7.

        Returns:
            RunRecord ready for JSON serialisation and persistence.
        """
        validity = self._unwrap(state.tool_results.get("ValidityTool", {}))
        sa = self._unwrap(state.tool_results.get("SAScoreTool", {}))
        sim = self._unwrap(state.tool_results.get("SimilarityTool", {}))

        return RunRecord(
            # ── Identity ────────────────────────────────────────────────
            run_id=str(uuid.uuid4()),
            molecule_id=state.molecule_id,
            run_timestamp=datetime.now(timezone.utc).isoformat(),

            # ── SMILES ──────────────────────────────────────────────────
            smiles_input=(
                str(state.raw_input) if isinstance(state.raw_input, str) else None
            ),
            smiles_canonical=validity.get("smiles_canonical"),

            # ── Validity ────────────────────────────────────────────────
            is_valid=validity.get("is_valid"),
            validity_error=validity.get("error_message"),
            num_atoms=validity.get("num_atoms"),
            num_bonds=validity.get("num_bonds"),

            # ── SA Score ────────────────────────────────────────────────
            sa_score=sa.get("sa_score"),
            sa_decision=sa.get("sa_decision"),
            synthesizability_category=sa.get("synthesizability_category"),
            sa_description=sa.get("sa_description"),
            sa_warning_flags=sa.get("warning_flags") or [],
            sa_complexity_breakdown=sa.get("complexity_breakdown"),
            sa_execution_time_ms=sa.get("execution_time_ms"),

            # ── Similarity ───────────────────────────────────────────────
            nn_tanimoto=sim.get("nearest_neighbor_tanimoto"),
            nn_source=sim.get("nearest_neighbor_source"),
            nn_id=sim.get("nearest_neighbor_id"),
            nn_name=sim.get("nearest_neighbor_name"),
            nn_smiles=sim.get("nearest_neighbor_smiles"),
            similarity_decision=sim.get("similarity_decision"),
            similarity_flag_threshold=sim.get("flag_threshold_used"),
            chembl_hit_count=len(sim.get("chembl_hits") or []),
            pubchem_hit_count=len(sim.get("pubchem_hits") or []),
            chembl_hits=sim.get("chembl_hits") or [],
            pubchem_hits=sim.get("pubchem_hits") or [],
            similarity_escalated=(
                (sim.get("nearest_neighbor_tanimoto") or 0.0) >= 0.95
                and sim.get("similarity_decision") == "FLAG"
            ),
            apis_queried=sim.get("apis_queried") or [],
            similarity_execution_time_ms=sim.get("execution_time_ms"),
            similarity_error=sim.get("error_reason"),

            # ── Final decision ───────────────────────────────────────────
            final_decision=explanation.decision,
            rule_rationale=(
                state.decision.rationale
                if state.decision and hasattr(state.decision, "rationale")
                else ""
            ),
            flags_raised=explanation.flags_raised,

            # ── Flat explanation ─────────────────────────────────────────
            explanation_summary=explanation.summary,
            explanation_sections=[s.to_dict() for s in explanation.sections],

            # ── AI reviewer ──────────────────────────────────────────────
            ai_review=ai_review,

            # ── Run metadata ─────────────────────────────────────────────
            early_termination=explanation.early_termination,
            termination_reason=explanation.termination_reason,
            total_execution_time_ms=self._compute_total_ms(state),
            tools_executed=list(state.tool_results.keys()),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(result: Any) -> Dict[str, Any]:
        """Normalise a tool result (plain dict or ToolResult dataclass) to dict."""
        if isinstance(result, dict):
            return result
        data = getattr(result, "data", None)
        if isinstance(data, dict):
            return data
        try:
            return dict(result)
        except (TypeError, ValueError):
            return {}

    @staticmethod
    def _compute_total_ms(state: Any) -> Optional[float]:
        """Compute wall-clock run time in ms from AgentState timestamps."""
        start = getattr(state, "execution_start_time", None)
        if start is None:
            return None
        end = datetime.now(timezone.utc)
        if isinstance(start, str):
            try:
                start = datetime.fromisoformat(start.replace("Z", "+00:00"))
            except ValueError:
                return None
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        return round((end - start).total_seconds() * 1000, 2)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save(record: RunRecord, path: str | Path) -> None:
    """
    Append a RunRecord as a single JSON line to a JSONL file.

    Creates the file and any parent directories if they do not exist.
    Uses an atomic rename strategy on the same filesystem to prevent
    partial writes from corrupting the log.

    Args:
        record: RunRecord to persist.
        path:   Path to the .jsonl file (e.g. "runs/triage_runs.jsonl").

    Raises:
        IOError: If the file cannot be written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(record.to_dict(), ensure_ascii=False, default=str)

    # Atomic append: write to a temp file then rename (same-dir, same-fs)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        # Read existing content
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        new_content = existing + line + "\n"
        tmp_path.write_text(new_content, encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        # Clean up temp file on failure
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    logger.debug(
        "RunRecord saved: run_id=%s molecule=%s decision=%s → %s",
        record.run_id, record.molecule_id, record.final_decision, path,
    )


def load_all(path: str | Path) -> List[RunRecord]:
    """
    Load all RunRecords from a JSONL file.

    Skips blank lines and lines that fail to parse, logging a warning
    for each bad line so the rest of the file is still accessible.

    Args:
        path: Path to the .jsonl file.

    Returns:
        List of RunRecord objects, in file order.
    """
    path = Path(path)
    if not path.exists():
        logger.info("load_all: file does not exist yet: %s", path)
        return []

    records: List[RunRecord] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(RunRecord.from_dict(data))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning(
                    "load_all: skipping malformed line %d in %s: %s",
                    lineno, path, exc,
                )

    logger.debug("load_all: loaded %d records from %s", len(records), path)
    return records


def load_as_dicts(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load all run records as plain dicts (no deserialisation overhead).

    Useful for pandas ingestion or ad-hoc analysis:
        import pandas as pd
        df = pd.DataFrame(load_as_dicts("runs/triage_runs.jsonl"))

    Args:
        path: Path to the .jsonl file.

    Returns:
        List of dicts, in file order. Malformed lines are skipped.
    """
    path = Path(path)
    if not path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(
                    "load_as_dicts: skipping malformed line %d in %s: %s",
                    lineno, path, exc,
                )
    return rows