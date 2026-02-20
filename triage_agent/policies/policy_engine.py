"""
policies/policy_engine.py

TRI_FLAG Policy Engine — governs triage routing decisions.

Week 3: Chemical validity check -> DISCARD if invalid
Week 4: SA score check -> DISCARD if SA > 7, FLAG if 6-7 (continues), PASS if < 6

ARCHITECTURE NOTE
-----------------
The existing triage_agent.py calls policy_engine.evaluate(state) and expects
a Decision object (from agent/decision.py) in return. This file preserves that
contract exactly — evaluate() signature and return type are unchanged from Week 3.

Internally, routing logic is handled by the new PolicyDecision class for
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
# Policy Protocol -- preserved from Week 3 (required by policies/__init__.py)
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
# PolicyDecision: internal routing result (Week 4 addition)
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
        1. ValidityTool  -- DISCARD if chemistry is invalid (Week 3)
        2. SAScoreTool   -- DISCARD if SA > flag_threshold (Week 4)
                         -- FLAG    if pass_threshold <= SA <= flag_threshold
                         -- PASS    if SA < pass_threshold
        3. Legacy pluggable policies (Week 3 design, still supported)
        4. Default fallback -- FLAG (conservative, no decisive result found)

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
            "PolicyEngine initialised -- %d policies, "
            "SA thresholds: pass=%.1f flag=%.1f",
            len(self.policies),
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

        Args:
            state: AgentState populated with tool results.

        Returns:
            Decision object (PASS / FLAG / DISCARD) with rationale.

        Raises:
            ValueError: If state is None or missing required attributes.
        """
        self._validate_state(state)

        # -- Week 3: validity check (highest priority) -------------------
        pd = self._check_validity(state)
        if pd is not None:
            self._log_provenance(state, pd)
            return pd.to_decision()

        # -- Week 4: SA score check --------------------------------------
        pd = self._check_sa_score(state)
        if pd is not None:
            self._log_provenance(state, pd)
            return pd.to_decision()

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

        # -- Default fallback --------------------------------------------
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

        # Handle ToolResult dataclass wrapper if present
        if hasattr(validity_result, 'data'):
            validity_result = validity_result.data or {}

        # Default False matches Week 3 exactly: missing key treated as invalid
        if not validity_result.get('is_valid', False):
            error_msg = validity_result.get('error_message', 'Unknown validation error')
            logger.warning(
                "[%s] Policy DISCARD: invalid chemistry -- %s",
                state.molecule_id, error_msg,
            )
            return PolicyDecision(
                decision="DISCARD",
                reason=f"Chemically invalid molecule: {error_msg}",
                tool_checked="ValidityTool",
                metadata={
                    'termination_reason': 'validity_check_failed',
                    'validity_error': error_msg,
                },
            )
        return None

    # ------------------------------------------------------------------
    # Week 4: SA score check
    # ------------------------------------------------------------------

    def _check_sa_score(self, state: AgentState) -> Optional[PolicyDecision]:
        sa_result = state.tool_results.get('SAScoreTool')
        if sa_result is None:
            return None

        # Handle ToolResult dataclass wrapper if present
        if hasattr(sa_result, 'data'):
            sa_result = sa_result.data or {}

        # Tool-level error
        if sa_result.get('sa_decision') == 'ERROR':
            error = sa_result.get('error_message', 'Unknown SA score error')
            logger.error(
                "[%s] Policy ERROR: SAScoreTool failed -- %s",
                state.molecule_id, error,
            )
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
            logger.warning(
                "[%s] Policy DISCARD: SA=%.2f (%s) > %.1f",
                state.molecule_id, sa_score, category,
                self.sa_thresholds.flag_threshold,
            )
            return PolicyDecision(
                decision="DISCARD",
                reason=description,
                tool_checked="SAScoreTool",
                metadata=metadata,
            )

        if tool_decision == 'FLAG':
            logger.warning(
                "[%s] Policy FLAG: SA=%.2f (%s) in [%.1f, %.1f] -- continuing",
                state.molecule_id, sa_score, category,
                self.sa_thresholds.pass_threshold,
                self.sa_thresholds.flag_threshold,
            )
            return PolicyDecision(
                decision="FLAG",
                reason=description,
                tool_checked="SAScoreTool",
                metadata=metadata,
            )

        # PASS
        return PolicyDecision(
            decision="PASS",
            reason=description,
            tool_checked="SAScoreTool",
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Helpers (preserved from Week 3, extended)
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
            state.identifier, len(state.tool_results)
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
            "Decision provenance: identifier=%s decision=%s tool=%s",
            state.identifier, pd.decision, pd.tool_checked,
        )

    def _log_decision_provenance(
        self,
        state: AgentState,
        decision: Decision,
        policy: Optional[Policy],
    ) -> None:
        provenance = {
            "identifier": state.identifier,
            "decision_action": decision.decision_type.name,
            "policy": policy.__class__.__name__ if policy else "fallback",
            "num_tool_results": len(state.tool_results),
            "tool_names": list(state.tool_results.keys()),
        }
        logger.info("Decision provenance: %s", provenance)


# ---------------------------------------------------------------------------
# PlaceholderPolicy -- preserved from Week 3 for backwards compatibility
# ---------------------------------------------------------------------------

class PlaceholderPolicy:
    """
    Placeholder policy demonstrating the pluggable policy interface.
    Always returns None (non-decisive). Preserved from Week 3.
    """

    def evaluate(self, state: AgentState) -> Optional[Decision]:
        logger.debug(
            "PlaceholderPolicy evaluated (non-decisive) for %s",
            state.identifier
        )
        return None