"""
policies/policy_engine.py

TRI_FLAG Policy Engine — governs triage routing decisions.

Week 3: Chemical validity check -> DISCARD if invalid
Week 4: SA score check -> DISCARD if SA > 7, FLAG if 6-7 (continues), PASS if < 6
         Added PolicyDecision, should_discard(), should_flag(), sa_thresholds param
Week 5: Similarity/IP-risk check -> FLAG on Tanimoto >= 0.85 (never DISCARD)
         Dual-source ChEMBL + PubChem reporting; escalation for Tanimoto >= 0.95
Week 6: Fix — SA PASS no longer accumulated as flag contributor in evaluate().
Week 9: Lipinski Ro5 checks from DescriptorTool results (advisory FLAG, no DISCARD).
        PAINS advisory FLAG from PAINSTool results (advisory FLAG, no DISCARD).
        Neither check changes DISCARD/PASS logic — they annotate state with FLAGS
        so the rationale text is more informative.

ARCHITECTURE NOTE
-----------------
The existing triage_agent.py calls policy_engine.evaluate(state) and expects
a Decision object (from agent/decision.py) in return. This file preserves that
contract exactly — evaluate() signature and return type are unchanged from Week 3.

Internally, routing logic is handled by the PolicyDecision class (Week 4) for
clarity, but evaluate() always converts to a Decision before returning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol

from agent.agent_state import AgentState
from agent.decision import Decision, DecisionType
from policies.thresholds import DEFAULT_SA_THRESHOLDS, SAScoreThresholds


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy Protocol -- preserved from Week 3
# ---------------------------------------------------------------------------

class Policy(Protocol):
    """
    Protocol defining the interface for individual policy rules.
    Preserved from Week 3 for backwards compatibility with policies/__init__.py
    and any code constructing PolicyEngine(policies=[...]).
    """

    def evaluate(self, state: AgentState) -> Optional[Decision]:
        """Evaluate agent state and optionally produce a decision."""
        ...


# ---------------------------------------------------------------------------
# PolicyDecision: internal routing result (Week 4, preserved exactly)
# ---------------------------------------------------------------------------

class PolicyDecision:
    """
    Internal structured result used within the policy engine.

    The public evaluate() method converts this to a Decision object to
    maintain full compatibility with the existing triage_agent.py interface.

    Attributes:
        decision:     "PASS" | "FLAG" | "DISCARD" | "ERROR"
        reason:       Human-readable explanation.
        tool_checked: Name of the tool whose result triggered this decision.
        metadata:     Optional supplementary data (sa_score, category, etc.)
    """
    __slots__ = ("decision", "reason", "tool_checked", "metadata")

    def __init__(
        self,
        decision: str,
        reason: str,
        tool_checked: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.decision = decision
        self.reason = reason
        self.tool_checked = tool_checked
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "tool_checked": self.tool_checked,
            "metadata": self.metadata,
        }

    def to_decision(self) -> Decision:
        """
        Convert to a Decision object for compatibility with triage_agent.py.

        Mapping:
            PASS    -> DecisionType.PASS
            FLAG    -> DecisionType.FLAG
            DISCARD -> DecisionType.DISCARD
            ERROR   -> DecisionType.DISCARD  (tool error treated as discard)
        """
        decision_map = {
            "PASS":    DecisionType.PASS,
            "FLAG":    DecisionType.FLAG,
            "DISCARD": DecisionType.DISCARD,
            "ERROR":   DecisionType.DISCARD,
        }
        decision_type = decision_map.get(self.decision, DecisionType.FLAG)
        return Decision(
            decision_type=decision_type,
            rationale=self.reason,
            metadata={
                "tool_checked": self.tool_checked,
                "policy_decision": self.decision,
                **self.metadata,
            },
        )

    def __repr__(self) -> str:
        return (
            f"PolicyDecision(decision={self.decision!r}, "
            f"tool_checked={self.tool_checked!r})"
        )


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """
    Evaluates tool results in AgentState and returns a Decision.

    Decision priority (checked in order):
        1. ValidityTool    -- DISCARD if chemistry is invalid (Week 3)
        2. SAScoreTool     -- DISCARD if SA > flag_threshold (Week 4)
                           -- FLAG    if pass_threshold <= SA <= flag_threshold
                           -- PASS    if SA < pass_threshold (not accumulated)
        3. SimilarityTool  -- FLAG on Tanimoto >= 0.85 (Week 5, never DISCARD)
        4. Legacy pluggable policies (Week 3 design, still supported)
        5. Default fallback -- FLAG (conservative, no decisive result found)

    Advisory annotations (Week 9, do not change routing outcome):
        - DescriptorTool: Lipinski Ro5 violations added as FLAGS to state
        - PAINSTool:      PAINS structural alerts added as FLAGS to state

    Returns a Decision object (from agent/decision.py). Interface is unchanged
    from Week 3 -- evaluate() signature and return type are identical.
    """

    def __init__(
        self,
        policies: Optional[List[Policy]] = None,
        default_action: DecisionType = DecisionType.FLAG,
        sa_thresholds: Optional[SAScoreThresholds] = None,
    ):
        # Original Week 3 params preserved exactly
        self.policies: List[Policy] = policies or []
        self.default_action: DecisionType = default_action
        # Week 4 addition: SA score thresholds
        self.sa_thresholds = sa_thresholds or DEFAULT_SA_THRESHOLDS

        logger.info(
            "PolicyEngine initialized: %d pluggable policies, default=%s, "
            "SA thresholds: pass=%.1f flag=%.1f",
            len(self.policies),
            default_action.name,
            self.sa_thresholds.pass_threshold,
            self.sa_thresholds.flag_threshold,
        )

    # ------------------------------------------------------------------
    # Public API -- returns Decision (interface unchanged from Week 3)
    # ------------------------------------------------------------------

    def evaluate(self, state: AgentState) -> Decision:
        """
        Evaluate tool results in state and return a triage Decision.

        Checks in priority order. Returns the first decisive result,
        or the default fallback if nothing is decisive.

        Week 6 fix: SA PASS results are returned immediately rather than
        accumulated. Only FLAG and DISCARD SA results are carried forward.
        This prevents a PASS SA score from being incorrectly composed with
        a similarity FLAG to produce a misleading "SA + Similarity" dual-flag.

        Week 9 addition: Lipinski and PAINS advisory checks run after the
        routing decision is determined. They annotate state with FLAGS but
        never override a PASS, FLAG, or DISCARD outcome.
        """
        self._validate_state(state)

        # ═══════════════════════════════════════════════════════════════
        # WEEK 9 ADDITION: Advisory checks — run first so FLAGS are in
        # state before the Decision is finalised and rationale is built.
        # These never change the routing outcome.
        # ═══════════════════════════════════════════════════════════════
        self._check_lipinski(state)
        self._check_pains(state)
        # ═══════════════════════════════════════════════════════════════
        # END WEEK 9 ADDITION
        # ═══════════════════════════════════════════════════════════════

        # -- Week 3: validity check (highest priority) -------------------
        pd = self._check_validity(state)
        if pd is not None:
            self._log_provenance(state, pd)
            return pd.to_decision()

        # -- Week 4: SA score check --------------------------------------
        pd = self._check_sa_score(state)
        if pd is not None:
            if pd.decision == "DISCARD":
                # Hard stop — molecule is unsynthesizable
                self._log_provenance(state, pd)
                return pd.to_decision()
            if pd.decision == "PASS":
                # SA is clean — do not accumulate, continue to similarity check
                accumulated_flag_pd = None
            else:
                # FLAG: SA is borderline — carry forward for possible composition
                accumulated_flag_pd = pd
        else:
            accumulated_flag_pd = None

        # -- Week 5: Similarity / IP risk check --------------------------
        sim_pd = self._check_similarity(state)
        if sim_pd is not None:
            if accumulated_flag_pd is not None:
                # Both SA and similarity flagged — compose into single dual-flag
                combined = self._compose_flags(accumulated_flag_pd, sim_pd, state)
                self._log_provenance(state, combined)
                return combined.to_decision()
            # Only similarity flagged — return directly
            accumulated_flag_pd = sim_pd

        # -- Legacy pluggable policies (Week 3 design) -------------------
        for idx, policy in enumerate(self.policies):
            logger.debug("Applying policy %d: %s", idx, policy.__class__.__name__)
            decision = policy.evaluate(state)
            if decision is not None:
                logger.info(
                    "Policy %s decisive for %s: %s",
                    policy.__class__.__name__,
                    state.identifier,
                    decision.decision_type.name,
                )
                self._log_decision_provenance(state, decision, policy)
                return decision

        # -- Return accumulated FLAG if any (SA-only or similarity-only) -
        if accumulated_flag_pd is not None:
            self._log_provenance(state, accumulated_flag_pd)
            return accumulated_flag_pd.to_decision()

        # -- SA was PASS and similarity was PASS — explicit PASS return ---
        # If we reach here with a completed SA PASS, we must return PASS.
        # The fallback default (FLAG) must not override a clean pipeline run.
        if pd is not None and pd.decision == "PASS":
            logger.info(
                "All checks passed for %s — returning PASS.", state.identifier
            )
            return pd.to_decision()

        # -- Default fallback (no tool results or inconclusive) ----------
        logger.warning(
            "No decisive policy for %s -- applying default: %s",
            state.identifier,
            self.default_action.name,
        )
        return self._create_fallback_decision(state)

    def should_discard(self, state: AgentState) -> bool:
        """Convenience: True if the current state should be discarded."""
        return self.evaluate(state).decision_type == DecisionType.DISCARD

    def should_flag(self, state: AgentState) -> bool:
        """Convenience: True if the current state should be flagged."""
        return self.evaluate(state).decision_type == DecisionType.FLAG

    def add_policy(self, policy: Policy) -> None:
        """Dynamically register a policy to the evaluation sequence."""
        self.policies.append(policy)
        logger.info("Added policy: %s", policy.__class__.__name__)

    def clear_policies(self) -> None:
        """Remove all registered policies."""
        n = len(self.policies)
        self.policies.clear()
        logger.warning("Cleared %d policies from engine", n)

    # ------------------------------------------------------------------
    # Week 3: Chemical validity check
    # ------------------------------------------------------------------

    def _check_validity(self, state: AgentState) -> Optional[PolicyDecision]:
        validity_result = state.tool_results.get('ValidityTool')
        if validity_result is None:
            return None

        if hasattr(validity_result, 'data'):
            validity_result = validity_result.data or {}

        if not validity_result.get('is_valid', False):
            error_msg = validity_result.get('error_message', 'Unknown validation error')
            logger.warning("[%s] Policy DISCARD: invalid chemistry -- %s", state.molecule_id, error_msg)
            return PolicyDecision(
                decision="DISCARD",
                reason=f"Chemically invalid molecule: {error_msg}",
                tool_checked="ValidityTool",
                metadata={'termination_reason': 'validity_check_failed', 'validity_error': error_msg},
            )
        return None

    # ------------------------------------------------------------------
    # Week 4: SA score check
    # ------------------------------------------------------------------

    def _check_sa_score(self, state: AgentState) -> Optional[PolicyDecision]:
        sa_result = state.tool_results.get('SAScoreTool')
        if sa_result is None:
            return None

        if hasattr(sa_result, 'data'):
            sa_result = sa_result.data or {}

        if sa_result.get('sa_decision') == 'ERROR':
            error = sa_result.get('error_message', 'Unknown SA score error')
            logger.error("[%s] Policy ERROR: SAScoreTool failed -- %s", state.molecule_id, error)
            return PolicyDecision(
                decision="ERROR",
                reason=f"SAScoreTool error: {error}",
                tool_checked="SAScoreTool",
                metadata={"error_message": error},
            )

        sa_score = sa_result.get('sa_score')
        if sa_score is None:
            return None

        tool_decision = sa_result.get('sa_decision', 'PASS')
        category = sa_result.get('synthesizability_category', 'unknown')
        description = sa_result.get('sa_description', f'SA score {sa_score:.2f}')
        warning_flags = sa_result.get('warning_flags', [])

        metadata = {
            'sa_score': sa_score,
            'synthesizability_category': category,
            'warning_flags': warning_flags,
        }

        if tool_decision == 'DISCARD':
            logger.warning("[%s] Policy DISCARD: SA=%.2f (%s)", state.molecule_id, sa_score, category)
            return PolicyDecision(
                decision="DISCARD",
                reason=description,
                tool_checked="SAScoreTool",
                metadata=metadata,
            )

        if tool_decision == 'FLAG':
            logger.warning(
                "[%s] Policy FLAG: SA=%.2f (%s) -- continuing",
                state.molecule_id, sa_score, category,
            )
            return PolicyDecision(
                decision="FLAG",
                reason=description,
                tool_checked="SAScoreTool",
                metadata=metadata,
            )

        # PASS — return so evaluate() can inspect pd.decision == "PASS"
        logger.info("[%s] SA score PASS: %.2f (%s)", state.molecule_id, sa_score, category)
        return PolicyDecision(
            decision="PASS",
            reason=description,
            tool_checked="SAScoreTool",
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Week 5: Similarity / IP-risk check
    # ------------------------------------------------------------------

    def _check_similarity(self, state: AgentState) -> Optional[PolicyDecision]:
        sim_result = state.tool_results.get("SimilarityTool")
        if sim_result is None:
            return None

        decision_str = sim_result.get("similarity_decision", "PASS")
        nn_tanimoto = sim_result.get("nearest_neighbor_tanimoto", 0.0)
        nn_source = sim_result.get("nearest_neighbor_source")
        nn_id = sim_result.get("nearest_neighbor_id")
        nn_name = sim_result.get("nearest_neighbor_name")
        chembl_hits = sim_result.get("chembl_hits", [])
        pubchem_hits = sim_result.get("pubchem_hits", [])
        threshold_used = sim_result.get("flag_threshold_used", 0.85)
        apis_queried = sim_result.get("apis_queried", [])

        if decision_str == "PASS":
            return None

        if decision_str == "ERROR":
            return PolicyDecision(
                decision="FLAG",
                reason=(
                    "Similarity screening could not complete (both ChEMBL and "
                    "PubChem APIs unavailable). IP risk unknown — flagging for manual review."
                ),
                tool_checked="SimilarityTool",
                metadata={
                    "similarity_decision": "ERROR",
                    "error_reason": sim_result.get("error_reason", "API unavailable"),
                },
            )

        rationale = self._build_similarity_rationale(
            nn_tanimoto=nn_tanimoto, nn_source=nn_source, nn_id=nn_id, nn_name=nn_name,
            chembl_hits=chembl_hits, pubchem_hits=pubchem_hits,
            threshold_used=threshold_used, apis_queried=apis_queried,
        )

        metadata: Dict[str, Any] = {
            "nearest_neighbor_tanimoto": nn_tanimoto,
            "nearest_neighbor_source": nn_source,
            "nearest_neighbor_id": nn_id,
            "nearest_neighbor_name": nn_name,
            "flag_threshold_used": threshold_used,
            "chembl_hits_count": len(chembl_hits),
            "pubchem_hits_count": len(pubchem_hits),
            "apis_queried": apis_queried,
        }

        if chembl_hits:
            best = max(chembl_hits, key=lambda h: h.get("tanimoto", 0.0))
            metadata["chembl_best_tanimoto"] = best.get("tanimoto", 0.0)
            metadata["chembl_best_id"] = best.get("id")

        if pubchem_hits:
            best = max(pubchem_hits, key=lambda h: h.get("tanimoto", 0.0))
            metadata["pubchem_best_tanimoto"] = best.get("tanimoto", 0.0)
            metadata["pubchem_best_id"] = best.get("id")

        if nn_tanimoto >= 0.95:
            metadata["escalated"] = True

        return PolicyDecision(
            decision="FLAG",
            reason=rationale,
            tool_checked="SimilarityTool",
            metadata=metadata,
        )

    def _build_similarity_rationale(
        self, nn_tanimoto, nn_source, nn_id, nn_name,
        chembl_hits, pubchem_hits, threshold_used, apis_queried,
    ) -> str:
        nn_label = nn_name or nn_id or "unknown compound"
        if nn_tanimoto >= 0.95:
            severity = "near-identical"
            advice = (
                "Priority IP review required — this may be an enantiomer, "
                "prodrug, or salt form of an existing drug or patent."
            )
        else:
            severity = "highly similar"
            advice = "IP review recommended — possible same scaffold as known compound."

        parts = [
            f"Similarity screening flagged molecule as {severity} to known compounds "
            f"(Tanimoto {nn_tanimoto:.3f} >= {threshold_used:.2f} threshold).",
            f"Nearest neighbor: {nn_label} [{nn_source}, {nn_id}].",
        ]

        if chembl_hits and pubchem_hits:
            bc = max(chembl_hits, key=lambda h: h.get("tanimoto", 0.0))
            bp = max(pubchem_hits, key=lambda h: h.get("tanimoto", 0.0))
            parts.append(
                f"Dual-source confirmation: ChEMBL best={bc.get('tanimoto',0.0):.3f} "
                f"({bc.get('id','?')}), PubChem best={bp.get('tanimoto',0.0):.3f} "
                f"({bp.get('id','?')})."
            )
        elif chembl_hits:
            parts.append(f"Source: ChEMBL ({len(chembl_hits)} hits above threshold).")
        elif pubchem_hits:
            parts.append(f"Source: PubChem ({len(pubchem_hits)} hits above threshold).")

        parts.append(advice)
        return " ".join(parts)

    def _compose_flags(
        self,
        sa_pd: PolicyDecision,
        sim_pd: PolicyDecision,
        state: AgentState,
    ) -> PolicyDecision:
        combined_reason = (
            f"Multiple concerns flagged for {state.identifier}. "
            f"(1) Synthetic accessibility: {sa_pd.reason} "
            f"(2) IP similarity: {sim_pd.reason}"
        )
        logger.info("Composed dual FLAG for %s: SA + Similarity", state.identifier)
        return PolicyDecision(
            decision="FLAG",
            reason=combined_reason,
            tool_checked="SAScoreTool+SimilarityTool",
            metadata={
                "flag_sources": ["SAScoreTool", "SimilarityTool"],
                "sa_flag_metadata": sa_pd.metadata,
                "similarity_flag_metadata": sim_pd.metadata,
            },
        )

    # ══════════════════════════════════════════════════════════════════════
    # WEEK 9 ADDITION: Advisory checks — Lipinski Ro5 and PAINS
    # These methods annotate state with advisory FLAGS. They never return
    # a PolicyDecision and never influence the routing outcome. Called at
    # the top of evaluate() so FLAGS appear in state before the Decision
    # is finalised and rationale is rendered.
    # ══════════════════════════════════════════════════════════════════════

    def _check_lipinski(self, state: AgentState) -> None:
        """
        Add advisory FLAGs for Lipinski rule-of-five violations.

        Thresholds: MW > 500 Da, logP > 5, HBD > 5, HBA > 10.
        These are informational only — a Lipinski violation does not
        change the PASS/FLAG/DISCARD outcome. It annotates the rationale
        so a medicinal chemist reviewing the output knows to investigate
        oral bioavailability.
        """
        desc = state.tool_results.get("DescriptorTool")
        if not desc:
            return

        # Unwrap ToolResult if needed
        if hasattr(desc, "data"):
            desc = desc.data or {}

        # Skip if descriptor calculation itself failed
        if desc.get("error_message"):
            return

        violations = []
        mw = desc.get("mol_weight")
        logp = desc.get("logp")
        hbd = desc.get("hbd")
        hba = desc.get("hba")

        if mw is not None and mw > 500:
            violations.append(f"MW={mw:.1f} Da (>500)")
        if logp is not None and logp > 5:
            violations.append(f"logP={logp:.2f} (>5)")
        if hbd is not None and hbd > 5:
            violations.append(f"HBD={hbd} (>5)")
        if hba is not None and hba > 10:
            violations.append(f"HBA={hba} (>10)")

        if violations:
            reason = "Lipinski Ro5 violation: " + "; ".join(violations)
            logger.info("[%s] Advisory FLAG: %s", state.molecule_id, reason)
            if hasattr(state, "add_flag"):
                state.add_flag(reason=reason, source="PolicyEngine")

    def _check_pains(self, state: AgentState) -> None:
        """
        Add advisory FLAG if PAINSTool detected a PAINS structural alert.

        PAINSTool.run() already calls state.add_flag() when it finds a match,
        so this method only fires if PAINSTool ran but its own flag was not
        added (e.g. if the tool is used without the agent pipeline). In normal
        pipeline operation this is a no-op — the flag is already in state.
        Kept here for completeness and for direct PolicyEngine usage outside
        the agent (e.g. batch scoring without TriageAgent).
        """
        pains = state.tool_results.get("PAINSTool")
        if not pains:
            return

        # Unwrap ToolResult if needed
        if hasattr(pains, "data"):
            pains = pains.data or {}

        if not pains.get("pains_alert", False):
            return

        # Check if flag already exists in state to avoid duplicates
        existing_flags = getattr(state, "_flags", [])
        already_flagged = any(
            "PAINS" in str(f) for f in existing_flags
        )
        if already_flagged:
            return

        matches = pains.get("pains_matches", [])
        summary = ", ".join(matches[:3]) if matches else "unknown pattern"
        reason = f"PAINS structural alert: {summary}"
        logger.info("[%s] Advisory FLAG (PolicyEngine): %s", state.molecule_id, reason)
        if hasattr(state, "add_flag"):
            state.add_flag(reason=reason, source="PolicyEngine")

    # ══════════════════════════════════════════════════════════════════════
    # END WEEK 9 ADDITION
    # ══════════════════════════════════════════════════════════════════════

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_state(self, state: AgentState) -> None:
        if state is None:
            raise ValueError("AgentState cannot be None")
        if not hasattr(state, 'identifier') or state.identifier is None:
            raise ValueError("AgentState must contain a valid identifier")
        if not hasattr(state, 'tool_results'):
            raise ValueError("AgentState must contain tool_results attribute")
        logger.debug(
            "State validation passed: %s, %d tool results",
            state.identifier, len(state.tool_results),
        )

    def _create_fallback_decision(self, state: AgentState) -> Decision:
        rationale = (
            f"No decisive policy triggered for {state.identifier}. "
            f"Applying conservative default action ({self.default_action.name}) "
            "pending further review or policy refinement."
        )
        return Decision(
            decision_type=self.default_action,
            rationale=rationale,
            metadata={
                "decision_type": "fallback",
                "num_policies_evaluated": len(self.policies),
                "tool_results_count": len(state.tool_results),
            },
        )

    def _log_provenance(self, state: AgentState, pd: PolicyDecision) -> None:
        logger.info(
            "Decision provenance: identifier=%s, action=%s, source=%s, tool_results=%s",
            state.identifier, pd.decision, pd.tool_checked, list(state.tool_results.keys()),
        )

    def _log_decision_provenance(
        self,
        state: AgentState,
        decision: Decision,
        policy: Optional[Policy],
    ) -> None:
        logger.info("Decision provenance: %s", {
            "identifier": state.identifier,
            "decision_action": decision.decision_type.name,
            "policy": policy.__class__.__name__ if policy else "fallback",
            "num_tool_results": len(state.tool_results),
            "tool_names": list(state.tool_results.keys()),
        })


# ---------------------------------------------------------------------------
# PlaceholderPolicy -- preserved from Week 3 for backwards compatibility
# ---------------------------------------------------------------------------

class PlaceholderPolicy:
    """Placeholder policy, always non-decisive. Preserved from Week 3."""

    def evaluate(self, state: AgentState) -> Optional[Decision]:
        logger.debug("PlaceholderPolicy: non-decisive for %s", state.identifier)
        return None