"""
reporting/rationale_builder.py

TRI_FLAG Week 9 — Flat Rationale Builder

Produces a structured, plain-English explanation of a triage decision by
reading directly from AgentState.tool_results. Intentionally decoupled from
PolicyEngine — the builder is a *reporting* layer, not decision logic.

Design goals:
    - Every line maps 1-to-1 to a concrete score or threshold check
    - Output is human-readable at a glance (terminal, log, Streamlit, PDF)
    - No business logic: builder never changes or re-derives the decision
    - Gracefully handles partial runs (early termination, API errors)

Public API:
    build(state: AgentState) -> TriageExplanation
    format_text(explanation: TriageExplanation) -> str
    format_dict(explanation: TriageExplanation) -> dict

Output structure (TriageExplanation):
    .molecule_id         str
    .smiles              str | None
    .decision            str          "PASS" | "FLAG" | "DISCARD"
    .summary             str          one-sentence plain-English verdict
    .sections            list[ExplanationSection]
        each section:
            .tool        str          "Validity" | "SA Score" | "Similarity"
                                      | "Descriptors" | "PAINS"
            .status      str          "PASS" | "FLAG" | "DISCARD" | "ERROR" | "SKIPPED"
            .headline    str          short verdict for this check
            .details     list[str]    supporting detail lines
    .flags_raised        list[dict]   raw flags from AgentState
    .early_termination   bool
    .termination_reason  str | None

Week 9 additions:
    _build_descriptor_section() — physicochemical property table
    _build_pains_section()      — PAINS structural alert list
    Both sections inserted into build() when validity passed.

Week 9 (similarity update):
    _build_similarity_section() updated for three-source architecture:
        ChEMBL     — flagging source (approved/bioactive drugs)
        SureChEMBL — flagging source (patent literature)
        PubChem    — informational only, rendered as a reference note,
                     never cited as the reason for a FLAG decision
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExplanationSection:
    """
    One tool's contribution to the overall explanation.

    Attributes:
        tool:     Display name of the tool (e.g. "SA Score").
        status:   Outcome of this check: PASS / FLAG / DISCARD / ERROR / SKIPPED.
        headline: Single-sentence verdict (e.g. "SA score 3.2 — easy to synthesize").
        details:  Supporting lines (thresholds used, nearest neighbor, etc.).
    """
    tool: str
    status: str
    headline: str
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "status": self.status,
            "headline": self.headline,
            "details": self.details,
        }


@dataclass
class TriageExplanation:
    """
    Complete flat explanation for a single triage run.

    This is the canonical output of RationaleBuilder.build(). It contains
    everything needed to render a human-readable report, a Streamlit card,
    a database row, or a JSON export — without further computation.

    Attributes:
        molecule_id:        Identifier passed to TriageAgent.run().
        smiles:             Canonical SMILES (None if validity failed).
        decision:           Final decision string: PASS | FLAG | DISCARD.
        summary:            One-sentence plain-English verdict.
        sections:           Per-tool explanation sections, in pipeline order.
        flags_raised:       Raw flag list from AgentState.get_flags().
        early_termination:  True if pipeline exited before all tools ran.
        termination_reason: Human-readable reason for early exit, or None.
    """
    molecule_id: str
    smiles: Optional[str]
    decision: str
    summary: str
    sections: List[ExplanationSection] = field(default_factory=list)
    flags_raised: List[Dict[str, str]] = field(default_factory=list)
    early_termination: bool = False
    termination_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "molecule_id": self.molecule_id,
            "smiles": self.smiles,
            "decision": self.decision,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections],
            "flags_raised": self.flags_raised,
            "early_termination": self.early_termination,
            "termination_reason": self.termination_reason,
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class RationaleBuilder:
    """
    Builds a TriageExplanation from a completed AgentState.

    Usage:
        builder = RationaleBuilder()
        explanation = builder.build(state)
        print(format_text(explanation))

    The builder reads only from state.tool_results and state.decision —
    it never re-runs chemistry logic or re-evaluates policies.
    """

    def build(self, state: Any) -> TriageExplanation:
        """
        Build a TriageExplanation from a finalized AgentState.

        Args:
            state: AgentState returned by TriageAgent.run().

        Returns:
            TriageExplanation with all sections populated.
        """
        decision_str = self._extract_decision_str(state)
        smiles = self._extract_canonical_smiles(state)
        flags = state.get_flags() if hasattr(state, "get_flags") else []
        early_term = state.is_terminated() if hasattr(state, "is_terminated") else False
        term_reason = (
            state.get_termination_reason()
            if hasattr(state, "get_termination_reason") else None
        )

        sections: List[ExplanationSection] = []
        sections.append(self._build_validity_section(state))

        # Only build downstream sections if validity passed
        validity_result = state.tool_results.get("ValidityTool", {})
        validity_data = self._unwrap_result(validity_result)
        if validity_data.get("is_valid", False):
            # ═══════════════════════════════════════════════════════════
            # WEEK 9 ADDITION: Descriptor section (position 2 in pipeline)
            # ═══════════════════════════════════════════════════════════
            sections.append(self._build_descriptor_section(state))
            # ═══════════════════════════════════════════════════════════
            sections.append(self._build_sa_section(state))
            sections.append(self._build_similarity_section(state))
            # ═══════════════════════════════════════════════════════════
            # WEEK 9 ADDITION: PAINS section (position 5 in pipeline)
            # ═══════════════════════════════════════════════════════════
            sections.append(self._build_pains_section(state))
            # ═══════════════════════════════════════════════════════════

        summary = self._compose_summary(state.molecule_id, decision_str, sections, flags)

        logger.debug(
            "RationaleBuilder: built explanation for %s → %s",
            state.molecule_id, decision_str,
        )

        return TriageExplanation(
            molecule_id=state.molecule_id,
            smiles=smiles,
            decision=decision_str,
            summary=summary,
            sections=sections,
            flags_raised=flags,
            early_termination=early_term,
            termination_reason=term_reason,
        )

    # ------------------------------------------------------------------
    # Section builders — one per tool
    # ------------------------------------------------------------------

    def _build_validity_section(self, state: Any) -> ExplanationSection:
        """Build the Validity check section."""
        result = state.tool_results.get("ValidityTool")
        if result is None:
            return ExplanationSection(
                tool="Validity",
                status="SKIPPED",
                headline="Validity check did not run.",
            )

        data = self._unwrap_result(result)
        is_valid: bool = data.get("is_valid", False)
        canonical: Optional[str] = data.get("smiles_canonical")
        num_atoms: int = data.get("num_atoms", 0)
        num_bonds: int = data.get("num_bonds", 0)
        error_msg: Optional[str] = data.get("error_message")

        if is_valid:
            headline = f"Valid SMILES — {num_atoms} atoms, {num_bonds} bonds."
            details = []
            if canonical:
                details.append(f"Canonical SMILES: {canonical}")
            return ExplanationSection(
                tool="Validity",
                status="PASS",
                headline=headline,
                details=details,
            )
        else:
            headline = "Invalid SMILES — molecule rejected."
            details = []
            if error_msg:
                details.append(f"RDKit error: {error_msg}")
            details.append("Pipeline terminated: no further checks performed.")
            return ExplanationSection(
                tool="Validity",
                status="DISCARD",
                headline=headline,
                details=details,
            )

    # ══════════════════════════════════════════════════════════════════════
    # WEEK 9 ADDITION: Descriptor section
    # ══════════════════════════════════════════════════════════════════════

    def _build_descriptor_section(self, state: Any) -> ExplanationSection:
        """
        Build the physicochemical descriptor section.

        Renders MW, logP, TPSA, HBD, HBA, rotatable bonds, and scaffold SMILES
        as a compact property table. Flags Lipinski violations inline.
        Shown only when validity passed and DescriptorTool ran.
        """
        result = state.tool_results.get("DescriptorTool")
        if result is None:
            return ExplanationSection(
                tool="Descriptors",
                status="SKIPPED",
                headline="Descriptor calculation did not run (Week 9 tool not yet registered).",
            )

        data = self._unwrap_result(result)
        error_msg: Optional[str] = data.get("error_message")

        if error_msg:
            return ExplanationSection(
                tool="Descriptors",
                status="ERROR",
                headline="Descriptor calculation failed — values unavailable.",
                details=[f"Error: {error_msg}"],
            )

        mw = data.get("mol_weight")
        logp = data.get("logp")
        tpsa = data.get("tpsa")
        hbd = data.get("hbd")
        hba = data.get("hba")
        rot = data.get("rotatable_bonds")
        scaffold = data.get("scaffold_smiles")

        # Check Lipinski violations for inline annotation
        violations = []
        if mw is not None and mw > 500:
            violations.append(f"MW={mw:.1f} Da exceeds 500")
        if logp is not None and logp > 5:
            violations.append(f"logP={logp:.2f} exceeds 5")
        if hbd is not None and hbd > 5:
            violations.append(f"HBD={hbd} exceeds 5")
        if hba is not None and hba > 10:
            violations.append(f"HBA={hba} exceeds 10")

        status = "FLAG" if violations else "PASS"
        headline = (
            "Lipinski Ro5 violation detected — oral bioavailability may be reduced."
            if violations else
            "Physicochemical properties calculated — Lipinski Ro5 compliant."
        )

        details = [
            f"  MW:              {mw:.3f} Da" if mw is not None else "  MW:              N/A",
            f"  logP:            {logp:.3f}" if logp is not None else "  logP:            N/A",
            f"  TPSA:            {tpsa:.3f} Å²" if tpsa is not None else "  TPSA:            N/A",
            f"  HBD / HBA:       {hbd} / {hba}" if (hbd is not None and hba is not None) else "  HBD / HBA:       N/A",
            f"  Rotatable bonds: {rot}" if rot is not None else "  Rotatable bonds: N/A",
            f"  Scaffold:        {scaffold}" if scaffold else "  Scaffold:        N/A",
        ]

        if violations:
            details.append("Lipinski violations:")
            for v in violations:
                details.append(f"  ⚠  {v}")

        exec_ms = data.get("execution_time_ms", 0.0)
        details.append(f"Computed in {exec_ms:.1f} ms.")

        return ExplanationSection(
            tool="Descriptors",
            status=status,
            headline=headline,
            details=details,
        )

    # ══════════════════════════════════════════════════════════════════════
    # END WEEK 9 ADDITION: Descriptor section
    # ══════════════════════════════════════════════════════════════════════

    def _build_sa_section(self, state: Any) -> ExplanationSection:
        """Build the SA Score section."""
        result = state.tool_results.get("SAScoreTool")
        if result is None:
            return ExplanationSection(
                tool="SA Score",
                status="SKIPPED",
                headline="SA score check did not run (pipeline terminated early).",
            )

        data = self._unwrap_result(result)
        sa_score: Optional[float] = data.get("sa_score")
        sa_decision: str = data.get("sa_decision", "ERROR")
        category: str = data.get("synthesizability_category", "unknown")
        description: str = data.get("sa_description", "")
        warning_flags: List[str] = data.get("warning_flags", [])
        error_msg: Optional[str] = data.get("error_message")
        exec_ms: float = data.get("execution_time_ms", 0.0)

        if sa_decision == "ERROR" or sa_score is None:
            return ExplanationSection(
                tool="SA Score",
                status="ERROR",
                headline="SA score calculation failed.",
                details=[f"Error: {error_msg}"] if error_msg else ["Unexpected computation error."],
            )

        score_str = f"{sa_score:.2f}"
        headline = f"SA score {score_str} ({category}) — {sa_decision}."

        details = []
        if description:
            details.append(description)

        details.append(
            f"Thresholds: PASS < 6.0, FLAG 6.0–7.0, DISCARD > 7.0  "
            f"(scale: 1 = trivially easy, 10 = practically impossible)"
        )

        if warning_flags:
            details.append("Complexity warnings:")
            for wf in warning_flags:
                details.append(f"  • {wf}")

        details.append(f"Computed in {exec_ms:.1f} ms.")

        return ExplanationSection(
            tool="SA Score",
            status=sa_decision,
            headline=headline,
            details=details,
        )

    def _build_similarity_section(self, state: Any) -> ExplanationSection:
        """
        Build the Similarity / IP-risk section.

        Three-source architecture (Week 9):
            ChEMBL     — flagging source, full hit details rendered
            SureChEMBL — flagging source, patent context rendered
            PubChem    — informational only, rendered as a reference note,
                         never cited as the reason for a FLAG decision
        """
        result = state.tool_results.get("SimilarityTool")
        if result is None:
            return ExplanationSection(
                tool="Similarity",
                status="SKIPPED",
                headline="Similarity check did not run (pipeline terminated early).",
            )

        data = self._unwrap_result(result)
        sim_decision: str        = data.get("similarity_decision", "ERROR")
        nn_tanimoto: float       = data.get("nearest_neighbor_tanimoto", 0.0)
        nn_source: Optional[str] = data.get("nearest_neighbor_source")
        nn_id: Optional[str]     = data.get("nearest_neighbor_id")
        nn_name: Optional[str]   = data.get("nearest_neighbor_name")
        nn_smiles: Optional[str] = data.get("nearest_neighbor_smiles")

        chembl_hits: List[dict]     = data.get("chembl_hits", [])
        surechembl_hits: List[dict] = data.get("surechembl_hits", [])
        pubchem_hits: List[dict]    = data.get("pubchem_hits", [])

        flag_threshold: float       = data.get("flag_threshold_used", 0.90)
        escalation_threshold: float = 0.95
        apis_queried: List[str]     = data.get("apis_queried", [])
        exec_ms: float              = data.get("execution_time_ms", 0.0)
        error_reason: Optional[str] = data.get("error_reason")

        # ── ERROR ─────────────────────────────────────────────────────────
        if sim_decision == "ERROR":
            return ExplanationSection(
                tool="Similarity",
                status="ERROR",
                headline="Similarity search failed — flagged conservatively.",
                details=[
                    f"Error: {error_reason}" if error_reason else "Flagging APIs unavailable.",
                    "Conservative FLAG applied: IP risk could not be confirmed as safe.",
                ],
            )

        # ── Headline ──────────────────────────────────────────────────────
        if sim_decision == "PASS":
            headline = (
                f"No similar compounds found in patent or drug databases "
                f"(best Tanimoto {nn_tanimoto:.3f} < {flag_threshold:.2f} threshold) — PASS."
            )
        else:
            escalated    = nn_tanimoto >= escalation_threshold
            severity     = "near-identical" if escalated else "similar"
            nn_label     = nn_name or nn_id or "unknown"
            source_label = nn_source or "unknown source"
            headline = (
                f"Flagged: {severity} to {nn_label} [{source_label}] "
                f"(Tanimoto {nn_tanimoto:.3f} >= {flag_threshold:.2f} threshold)."
            )

        details = []

        # ── Nearest neighbour (flagging sources only) ─────────────────────
        if nn_tanimoto > 0.0 and nn_id:
            nn_line = (
                f"Nearest neighbor: {nn_name or 'unnamed'} | "
                f"ID: {nn_id} | Source: {nn_source or '?'} | "
                f"Tanimoto: {nn_tanimoto:.3f}"
            )
            details.append(nn_line)
            if nn_smiles:
                details.append(f"  NN SMILES: {nn_smiles}")

        # ── ChEMBL hits ───────────────────────────────────────────────────
        if chembl_hits:
            best_chembl = max(chembl_hits, key=lambda h: h.get("tanimoto", 0.0))
            details.append(
                f"ChEMBL: {len(chembl_hits)} hit(s) above threshold "
                f"(best: {best_chembl.get('tanimoto', 0.0):.3f}, ID: {best_chembl.get('id', '?')})"
            )

        # ── SureChEMBL hits ───────────────────────────────────────────────
        if surechembl_hits:
            best_sc    = max(surechembl_hits, key=lambda h: h.get("tanimoto", 0.0))
            patent_ids = best_sc.get("patent_ids", [])
            patent_note = f", patent: {patent_ids[0]}" if patent_ids else ""
            details.append(
                f"SureChEMBL: {len(surechembl_hits)} hit(s) in patent literature "
                f"(best: {best_sc.get('tanimoto', 0.0):.3f}, "
                f"ID: {best_sc.get('id', '?')}{patent_note})"
            )

        # ── PubChem — informational note only ─────────────────────────────
        if pubchem_hits:
            details.append(
                f"PubChem reference: {len(pubchem_hits)} structurally related "
                f"compound(s) found (informational only — not used for FLAG decision)."
            )

        # ── Threshold context ─────────────────────────────────────────────
        details.append(
            f"FLAG sources: ChEMBL (approved drugs) and SureChEMBL (patent literature). "
            f"Threshold: Tanimoto >= {flag_threshold:.2f}. "
            f"Escalated (near-identical): >= {escalation_threshold:.2f}."
        )
        if nn_tanimoto >= escalation_threshold and sim_decision == "FLAG":
            details.append(
                "⚠  ESCALATED: Tanimoto >= 0.95 indicates near-identical structure. "
                "Priority IP review required."
            )

        if apis_queried:
            details.append(f"APIs queried: {', '.join(apis_queried)}")
        details.append(f"Completed in {exec_ms:.1f} ms.")

        return ExplanationSection(
            tool="Similarity",
            status=sim_decision,
            headline=headline,
            details=details,
        )

    # ══════════════════════════════════════════════════════════════════════
    # WEEK 9 ADDITION: PAINS section
    # ══════════════════════════════════════════════════════════════════════

    def _build_pains_section(self, state: Any) -> ExplanationSection:
        """
        Build the PAINS structural alert section.

        Renders matched PAINS pattern names with a one-line explanation of
        what PAINS means. Advisory only — never causes DISCARD.
        Shown only when validity passed and PAINSTool ran.
        """
        result = state.tool_results.get("PAINSTool")
        if result is None:
            return ExplanationSection(
                tool="PAINS",
                status="SKIPPED",
                headline="PAINS screen did not run (Week 9 tool not yet registered).",
            )

        data = self._unwrap_result(result)
        error_msg: Optional[str] = data.get("error_message")

        if error_msg:
            return ExplanationSection(
                tool="PAINS",
                status="ERROR",
                headline="PAINS screen failed — structural alert status unknown.",
                details=[f"Error: {error_msg}"],
            )

        pains_alert: bool = data.get("pains_alert", False)
        pains_matches: List[str] = data.get("pains_matches", [])
        exec_ms: float = data.get("execution_time_ms", 0.0)

        if not pains_alert:
            return ExplanationSection(
                tool="PAINS",
                status="PASS",
                headline="No PAINS structural alerts detected.",
                details=[
                    "PAINS (Pan-Assay Interference Compounds) are motifs known to cause "
                    "false positives in biochemical assays. None matched.",
                    f"Screened in {exec_ms:.1f} ms.",
                ],
            )

        # PAINS matched — build advisory FLAG section
        headline = (
            f"PAINS alert: {len(pains_matches)} structural pattern(s) matched — advisory FLAG."
        )

        details = [
            "PAINS (Pan-Assay Interference Compounds) are structural motifs known to",
            "cause false positives in biochemical assays due to reactivity, fluorescence",
            "interference, or aggregation. Not disqualifying — flag for wet-lab follow-up.",
            "Matched patterns:",
        ]
        for pattern in pains_matches:
            details.append(f"  • {pattern}")

        details.append(
            "Recommendation: confirm activity with orthogonal assay before committing "
            "to this scaffold."
        )
        details.append(f"Screened in {exec_ms:.1f} ms.")

        return ExplanationSection(
            tool="PAINS",
            status="FLAG",
            headline=headline,
            details=details,
        )

    # ══════════════════════════════════════════════════════════════════════
    # END WEEK 9 ADDITION: PAINS section
    # ══════════════════════════════════════════════════════════════════════

    # ------------------------------------------------------------------
    # Summary composition
    # ------------------------------------------------------------------

    def _compose_summary(
        self,
        molecule_id: str,
        decision: str,
        sections: List[ExplanationSection],
        flags: List[Dict[str, str]],
    ) -> str:
        """Compose a single-sentence plain-English summary of the outcome."""
        statuses = {s.tool: s.status for s in sections}

        if decision == "DISCARD":
            if statuses.get("Validity") in ("DISCARD", "ERROR"):
                return (
                    f"{molecule_id} was discarded: SMILES string is chemically invalid."
                )
            sa_status = statuses.get("SA Score", "")
            if sa_status == "DISCARD":
                sa_data = next(
                    (s for s in sections if s.tool == "SA Score"), None
                )
                score_hint = ""
                if sa_data and sa_data.details:
                    score_hint = f" ({sa_data.headline})"
                return (
                    f"{molecule_id} was discarded: too difficult to synthesize{score_hint}."
                )
            return f"{molecule_id} was discarded by policy evaluation."

        if decision == "FLAG":
            flag_sources = [f["source"] for f in flags] if flags else []
            # Check for PAINS in summary
            pains_flagged = "PAINSTool" in flag_sources or statuses.get("PAINS") == "FLAG"
            lipinski_flagged = any("Lipinski" in str(f.get("reason", "")) for f in flags)

            if "SAScoreTool" in flag_sources and "SimilarityTool" in flag_sources:
                return (
                    f"{molecule_id} was flagged for human review: "
                    f"borderline synthesizability AND IP similarity concerns."
                )
            if "SimilarityTool" in flag_sources:
                sim_section = next(
                    (s for s in sections if s.tool == "Similarity"), None
                )
                hint = f" ({sim_section.headline})" if sim_section else ""
                return f"{molecule_id} was flagged for IP review{hint}."
            if "SAScoreTool" in flag_sources:
                return f"{molecule_id} was flagged: SA score in borderline range (6.0–7.0)."
            if pains_flagged:
                return (
                    f"{molecule_id} was flagged: PAINS structural alert detected — "
                    f"confirm activity with orthogonal assay."
                )
            if lipinski_flagged:
                return (
                    f"{molecule_id} was flagged: Lipinski rule-of-five violation — "
                    f"oral bioavailability may be reduced."
                )
            return f"{molecule_id} was flagged for human review."

        if decision == "PASS":
            return (
                f"{molecule_id} passed all checks: "
                f"valid structure, acceptable synthesizability, no IP similarity concerns."
            )

        return f"{molecule_id} outcome: {decision}."

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_decision_str(state: Any) -> str:
        """Extract decision string from state, defaulting to 'UNKNOWN'."""
        decision = getattr(state, "decision", None)
        if decision is None:
            return "UNKNOWN"
        dt = getattr(decision, "decision_type", None)
        if dt is None:
            return "UNKNOWN"
        return dt.name if hasattr(dt, "name") else str(dt).upper()

    @staticmethod
    def _extract_canonical_smiles(state: Any) -> Optional[str]:
        """Extract canonical SMILES from ValidityTool result, or fall back to raw_input."""
        validity = state.tool_results.get("ValidityTool", {})
        data = RationaleBuilder._unwrap_result(validity)
        canonical = data.get("smiles_canonical")
        if canonical:
            return canonical
        raw = getattr(state, "raw_input", None)
        return str(raw) if isinstance(raw, str) else None

    @staticmethod
    def _unwrap_result(result: Any) -> Dict[str, Any]:
        """
        Normalize a tool result to a plain dict.

        Tool results may be stored as plain dicts or as ToolResult dataclasses
        (which have a .data attribute). This helper handles both.
        """
        if isinstance(result, dict):
            return result
        # ToolResult dataclass wraps the payload in .data
        data = getattr(result, "data", None)
        if isinstance(data, dict):
            return data
        # Last resort: try converting directly
        try:
            return dict(result)
        except (TypeError, ValueError):
            return {}


# ---------------------------------------------------------------------------
# Module-level formatting functions
# ---------------------------------------------------------------------------

def format_text(explanation: TriageExplanation) -> str:
    """
    Render a TriageExplanation as a clean terminal/log-friendly text block.

    Example output:
        ══════════════════════════════════════════════════
        TRI_FLAG TRIAGE REPORT
        ══════════════════════════════════════════════════
        Molecule  : mol_001
        SMILES    : CCO
        Decision  : FLAG
        Summary   : mol_001 was flagged for IP review ...
        ──────────────────────────────────────────────────
        [PASS]  Validity
                Valid SMILES — 3 atoms, 2 bonds.
        ...
        ══════════════════════════════════════════════════
    """
    lines: List[str] = []
    WIDE = 58
    THIN = 58

    lines.append("═" * WIDE)
    lines.append("  TRI_FLAG TRIAGE REPORT")
    lines.append("═" * WIDE)
    lines.append(f"  Molecule  : {explanation.molecule_id}")
    if explanation.smiles:
        lines.append(f"  SMILES    : {explanation.smiles}")
    lines.append(f"  Decision  : {explanation.decision}")
    lines.append(f"  Summary   : {explanation.summary}")

    if explanation.early_termination and explanation.termination_reason:
        lines.append(f"  ⚠  Early exit : {explanation.termination_reason}")

    lines.append("─" * THIN)

    for section in explanation.sections:
        status_badge = f"[{section.status:<7}]"
        lines.append(f"  {status_badge}  {section.tool}")
        lines.append(f"           {section.headline}")
        for detail in section.details:
            lines.append(f"             {detail}")
        lines.append("")

    if explanation.flags_raised:
        lines.append("─" * THIN)
        lines.append("  FLAGS RAISED:")
        for flag in explanation.flags_raised:
            lines.append(f"    • [{flag.get('source', '?')}] {flag.get('reason', '')}")
        lines.append("")

    lines.append("═" * WIDE)
    return "\n".join(lines)


def format_dict(explanation: TriageExplanation) -> Dict[str, Any]:
    """
    Render a TriageExplanation as a plain dict suitable for JSON serialisation.
    """
    return explanation.to_dict()