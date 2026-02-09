"""
PolicyEngine: Core decision-making component for TRI_FLAG agentic research system.

This module provides deterministic, explainable triage decisions for molecular
screening workflows. The PolicyEngine interprets accumulated tool outputs and
produces structured decisions (PASS/FLAG/DISCARD) with human-readable rationales.

Design Principles:
    - Deterministic: Same state always produces same decision
    - Inspectable: Decision provenance is explicitly traceable
    - Modular: Policies are pluggable and composable
    - Domain-agnostic: Thresholds and rules are externalized
    - Fail-explicit: Uncertainty is surfaced, not hidden

Author: [Your Name/Team]
License: [Your License]
"""

from typing import List, Optional, Protocol
from dataclasses import dataclass
import logging

from agent.agent_state import AgentState
from agent.decision import Decision, DecisionType


# Configure module logger
logger = logging.getLogger(__name__)


# ============================================================================
# Policy Protocol (Extensibility Interface)
# ============================================================================

class Policy(Protocol):
    """
    Protocol defining the interface for individual policy rules.
    
    Policies are composable evaluation units that inspect AgentState and
    optionally produce a Decision. Policies should be:
        - Stateless (operate only on provided state)
        - Deterministic (no internal randomness)
        - Self-documenting (clear rationale in decisions)
    
    Returns:
        Optional[Decision]: A decision if this policy is decisive,
                           None if evaluation should continue to next policy.
    """
    
    def evaluate(self, state: AgentState) -> Optional[Decision]:
        """
        Evaluate agent state and optionally produce a decision.
        
        Args:
            state: Current agent state containing tool results and metadata
            
        Returns:
            Decision object if this policy reaches a conclusion, else None
        """
        ...


# ============================================================================
# PolicyEngine: Primary Decision Component
# ============================================================================

class PolicyEngine:
    """
    Evaluates accumulated agent state and produces triage decisions.
    
    The PolicyEngine is the authoritative component for converting raw tool
    outputs and agent observations into actionable molecular triage outcomes.
    It operates as a deterministic interpreter, not a learning system.
    
    Architecture:
        1. Accepts AgentState (accumulated evidence from tools)
        2. Applies ordered sequence of Policy rules
        3. Returns structured Decision with rationale
    
    The engine supports:
        - Sequential rule evaluation (first decisive policy wins)
        - Pluggable policy registration
        - Explicit handling of incomplete evidence
        - Full provenance tracking for scientific reproducibility
    
    The engine does NOT:
        - Execute tools or interact with external systems
        - Perform action selection or planning
        - Use probabilistic or learned models (at this stage)
    
    Typical Usage:
        >>> engine = PolicyEngine(policies=[SafetyPolicy(), NoveltyPolicy()])
        >>> decision = engine.evaluate(agent_state)
        >>> print(f"{decision.decision_type}: {decision.rationale}")
    
    Attributes:
        policies: Ordered list of Policy objects to evaluate
        default_action: Action to take if no policy is decisive
    """
    
    def __init__(
        self,
        policies: Optional[List[Policy]] = None,
        default_action: DecisionType = DecisionType.FLAG
    ):
        """
        Initialize the PolicyEngine with an ordered policy sequence.
        
        Args:
            policies: List of Policy objects to evaluate in order.
                     If None, uses empty list (all molecules flagged by default).
            default_action: Decision action if no policy is decisive.
                           Defaults to FLAG (conservative fallback).
        
        Raises:
            ValueError: If default_action is invalid
        """
        self.policies: List[Policy] = policies or []
        self.default_action: DecisionType = default_action
        
        # Validate default action
        if not isinstance(default_action, DecisionType):
            raise ValueError(
                f"Invalid default_action: {default_action}. "
                f"Must be a DecisionType enum member."
            )
        
        logger.info(
            f"PolicyEngine initialized with {len(self.policies)} policies, "
            f"default_action={default_action.name}"
        )
    
    def evaluate(self, state: AgentState) -> Decision:
        """
        Primary evaluation entry point: convert AgentState to Decision.
        
        Evaluation Flow:
            1. Validate input state
            2. Iterate through registered policies in order
            3. Return first decisive policy result
            4. If no policy is decisive, apply default fallback
            5. Log decision provenance for reproducibility
        
        Args:
            state: AgentState containing tool results and execution metadata
        
        Returns:
            Decision object with action (PASS/FLAG/DISCARD) and rationale
        
        Raises:
            ValueError: If state is invalid or missing required fields
        
        Notes:
            - This method is deterministic: same state → same decision
            - Policy order matters: first match wins
            - Default fallback ensures a decision is always produced
        """
        # Step 1: Validate input state
        self._validate_state(state)
        
        logger.debug(f"Evaluating state for identifier: {state.identifier}")
        
        # Step 2: Sequential policy evaluation
        for idx, policy in enumerate(self.policies):
            logger.debug(f"Applying policy {idx}: {policy.__class__.__name__}")
            
            decision = policy.evaluate(state)
            
            if decision is not None:
                # Decisive policy found
                logger.info(
                    f"Policy {policy.__class__.__name__} produced decision: "
                    f"{decision.decision_type.name}"
                )
                self._log_decision_provenance(state, decision, policy)
                return decision
        
        # Step 3: No policy was decisive - apply default fallback
        logger.warning(
            f"No policy decisive for {state.identifier}. "
            f"Applying default action: {self.default_action.name}"
        )
        
        fallback_decision = self._create_fallback_decision(state)
        self._log_decision_provenance(state, fallback_decision, policy=None)
        
        return fallback_decision
    
    def _validate_state(self, state: AgentState) -> None:
        """
        Validate that AgentState contains minimum required information.
        
        Args:
            state: AgentState to validate
        
        Raises:
            ValueError: If state is missing required fields or is malformed
        """
        if state is None:
            raise ValueError("AgentState cannot be None")
        
        if not hasattr(state, 'identifier') or state.identifier is None:
            raise ValueError("AgentState must contain a valid identifier")
        
        if not hasattr(state, 'tool_results'):
            raise ValueError("AgentState must contain tool_results attribute")
        
        logger.debug(
            f"State validation passed: {state.identifier}, "
            f"{len(state.tool_results)} tool results"
        )
    
    def _create_fallback_decision(self, state: AgentState) -> Decision:
        """
        Create a default decision when no policy is decisive.
        
        This is a conservative fallback that ensures the system always
        produces a decision, even when evidence is insufficient or ambiguous.
        
        Args:
            state: AgentState that did not trigger any policy
        
        Returns:
            Decision with default action and explanatory rationale
        """
        rationale = (
            f"No decisive policy triggered for {state.identifier}. "
            f"Applying conservative default action ({self.default_action.name}) "
            f"pending further review or policy refinement."
        )
        
        metadata = {
            "decision_type": "fallback",
            "num_policies_evaluated": len(self.policies),
            "tool_results_count": len(state.tool_results)
        }
        
        return Decision(
            decision_type=self.default_action,
            rationale=rationale,
            metadata=metadata
        )
    
    def _log_decision_provenance(
        self,
        state: AgentState,
        decision: Decision,
        policy: Optional[Policy]
    ) -> None:
        """
        Log detailed provenance information for reproducibility.
        
        This method creates an audit trail linking decisions back to
        the specific state and policy that produced them.
        
        Args:
            state: AgentState that was evaluated
            decision: Decision that was produced
            policy: Policy that produced the decision (None if fallback)
        """
        provenance = {
            "identifier": state.identifier,
            "decision_action": decision.decision_type.name,
            "policy": policy.__class__.__name__ if policy else "fallback",
            "num_tool_results": len(state.tool_results),
            "tool_names": list(state.tool_results.keys())
        }
        
        logger.info(f"Decision provenance: {provenance}")
    
    def add_policy(self, policy: Policy) -> None:
        """
        Dynamically register a new policy to the evaluation sequence.
        
        The policy will be appended to the end of the current policy list.
        For insertion at specific positions, directly modify self.policies.
        
        Args:
            policy: Policy object implementing the Policy protocol
        
        Example:
            >>> engine.add_policy(CustomSafetyPolicy())
        """
        self.policies.append(policy)
        logger.info(f"Added policy: {policy.__class__.__name__}")
    
    def clear_policies(self) -> None:
        """
        Remove all registered policies.
        
        After calling this method, the engine will always apply the
        default fallback action.
        """
        num_removed = len(self.policies)
        self.policies.clear()
        logger.warning(f"Cleared {num_removed} policies from engine")


# ============================================================================
# Example Placeholder Policy (Demonstrates Extension Pattern)
# ============================================================================

@dataclass
class PlaceholderPolicy:
    """
    Placeholder policy demonstrating the extension interface.
    
    This policy always returns None (non-decisive), serving as a
    template for implementing real domain-specific policies.
    
    Future policies should:
        - Implement the Policy protocol
        - Inspect relevant tool results from state.tool_results
        - Apply deterministic rule logic
        - Return Decision with clear rationale
        - Return None if unable to reach conclusion
    
    Example future policy types:
        - ThresholdSafetyPolicy: Check toxicity scores against limits
        - StructuralFilterPolicy: PAINS filter, reactive group detection
        - NoveltyPolicy: Similarity search against known compounds
        - CompletenessPolicy: Verify minimum required data present
    """
    
    def evaluate(self, state: AgentState) -> Optional[Decision]:
        """
        Placeholder evaluation (always non-decisive).
        
        Real implementations should:
            1. Extract relevant tool results from state
            2. Apply domain-specific logic
            3. Construct Decision if criteria met
            4. Return None if evaluation should continue
        
        Example pattern:
```
            toxicity_result = state.tool_results.get('toxicity_predictor')
            if toxicity_result and toxicity_result['score'] > threshold:
                return Decision(
                    action=DecisionType.FLAG,
                    rationale=f"Toxicity score {score} exceeds threshold",
                    metadata={'toxicity_score': score}
                )
            return None
```
        """
        logger.debug(
            f"PlaceholderPolicy evaluated (non-decisive) for "
            f"{state.identifier}"
        )
        return None


# ============================================================================
# Future Extensibility Hooks (Documentation)
# ============================================================================

"""
EXTENSIBILITY ROADMAP
=====================

This PolicyEngine is designed to support future enhancements without
architectural changes:

1. CONFIG-DRIVEN THRESHOLDS
   - Define PolicyConfig dataclass with threshold parameters
   - Pass config to policy constructors
   - Load from YAML/JSON configuration files
   
   Example:
       @dataclass
       class SafetyPolicyConfig:
           toxicity_threshold: float = 0.7
           reactivity_threshold: float = 0.5
       
       policy = SafetyPolicy(config=SafetyPolicyConfig.from_yaml('config.yaml'))

2. COMPOSITE POLICY RULES
   - Implement AndPolicy, OrPolicy for logical composition
   - Allow nested policy trees
   
   Example:
       composite = AndPolicy([
           SafetyPolicy(),
           OrPolicy([NoveltyPolicy(), RarityPolicy()])
       ])

3. EXPLANATION MODULES
   - Add optional LLM-based explanation layer
   - LLM interprets Decision rationale, does NOT make decisions
   - Useful for grant reports, stakeholder communication
   
   Example:
       explainer = LLMExplainer(model='claude-sonnet-4')
       verbose_rationale = explainer.expand(decision.rationale, state)

4. POLICY METADATA & VERSIONING
   - Add version strings to policies
   - Track which policy version produced each decision
   - Support reproducibility across code updates

5. WEIGHTED VOTING (Advanced)
   - If needed, implement PolicyAggregator
   - Collect scores from multiple policies
   - Apply deterministic aggregation function
   - Still maintains full determinism and provenance

All extensions should preserve:
    - Determinism (same state → same decision)
    - Inspectability (decision provenance is traceable)
    - Modularity (policies remain independent)
"""
