"""
agent/decision.py

This module defines the core decision data structures that represent the outcome
of an agent's evaluation of a candidate.

Design Philosophy:
-----------------
Decisions are immutable, explicit data objects that capture:
1. What decision was made (pass/flag/discard)
2. Why the decision was made (rationale)
3. Supporting context (metadata)

This separation of decision representation from decision logic enables:
- Clear audit trails
- Systematic evaluation
- Transparent explanations to users
- Easy serialization for storage/display

Decisions flow through the system as follows:
1. Policy logic evaluates candidate → creates Decision
2. Decision stored in AgentState
3. UI/reporting layers consume Decision for display
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime


class DecisionType(Enum):
    """
    Enumeration of possible agent decisions.
    
    Each decision type represents a terminal state for candidate evaluation:
    
    PASS: 
        Candidate meets all criteria and should proceed.
        Typically means no red flags detected, all requirements satisfied.
        
    FLAG:
        Candidate has concerns that require human review.
        Not automatically rejected, but needs manual judgment.
        Example: Ambiguous information, borderline criteria, conflicting signals.
        
    DISCARD:
        Candidate definitively fails criteria and should not proceed.
        Clear policy violations or missing critical requirements.
    """
    PASS = "pass"
    FLAG = "flag"
    DISCARD = "discard"
    
    def __str__(self) -> str:
        """String representation for logging and display."""
        return self.value
    
    def is_passing(self) -> bool:
        """Helper: Check if this decision allows candidate to proceed."""
        return self == DecisionType.PASS
    
    def requires_review(self) -> bool:
        """Helper: Check if this decision requires human review."""
        return self == DecisionType.FLAG


@dataclass
class Decision:
    """
    Structured representation of an agent's evaluation outcome.
    
    This is the primary output artifact of agent reasoning. It captures not just
    the final decision, but the reasoning process and supporting evidence.
    
    Attributes:
    ----------
    decision_type : DecisionType
        The categorical outcome (PASS/FLAG/DISCARD). This is the "what" of the decision.
        
    rationale : str
        Human-readable explanation of why this decision was made. This should be
        clear, specific, and actionable. Good rationales reference specific evidence
        and policy rules.
        
        Examples:
        - "Candidate has 5+ years Python experience, exceeding 3-year requirement"
        - "Resume contains potential misrepresentation: claims PhD from non-accredited institution"
        - "Missing required security clearance documentation"
        
    metadata : Dict[str, Any]
        Structured supporting data for the decision. This enables programmatic
        analysis and provides context for human review.
        
        Common metadata fields:
        - "confidence": float (0.0-1.0) indicating decision certainty
        - "evidence": List of specific facts that supported the decision
        - "policy_violations": List of policy IDs that were violated
        - "scores": Dict of sub-scores (e.g., technical: 0.8, experience: 0.6)
        - "timestamp": When the decision was made
        - "agent_version": Which version of agent logic produced this
        
    Examples:
    --------
    # Straightforward pass
    Decision(
        decision_type=DecisionType.PASS,
        rationale="Candidate meets all technical requirements with strong background",
        metadata={
            "confidence": 0.95,
            "technical_score": 0.9,
            "experience_years": 7
        }
    )
    
    # Flag for review
    Decision(
        decision_type=DecisionType.FLAG,
        rationale="Employment gap of 18 months requires explanation",
        metadata={
            "confidence": 0.7,
            "gap_start": "2022-03",
            "gap_end": "2023-09",
            "requires_followup": True
        }
    )
    
    # Clear discard
    Decision(
        decision_type=DecisionType.DISCARD,
        rationale="Candidate lacks required certifications (PMP, CISSP)",
        metadata={
            "confidence": 1.0,
            "missing_requirements": ["PMP", "CISSP"],
            "policy_id": "REQ-2024-007"
        }
    )
    """
    
    decision_type: DecisionType
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """
        Validation and normalization after initialization.
        
        Ensures:
        - decision_type is actually a DecisionType enum
        - rationale is non-empty
        - metadata is a dictionary
        
        This prevents malformed Decision objects from propagating through the system.
        """
        # Ensure decision_type is a DecisionType enum
        if not isinstance(self.decision_type, DecisionType):
            raise TypeError(
                f"decision_type must be DecisionType enum, got {type(self.decision_type)}"
            )
        
        # Ensure rationale is provided and non-empty
        if not self.rationale or not self.rationale.strip():
            raise ValueError("rationale cannot be empty - decisions must be explained")
        
        # Ensure metadata is a dictionary
        if not isinstance(self.metadata, dict):
            raise TypeError(f"metadata must be dict, got {type(self.metadata)}")
        
        # Automatically add timestamp if not present
        # This enables temporal analysis of decisions
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Decision to dictionary for serialization.
        
        Useful for:
        - JSON serialization for API responses
        - Database storage
        - Logging
        
        Returns:
        -------
        Dict containing all decision data with enum converted to string.
        """
        return {
            "decision_type": self.decision_type.value,
            "rationale": self.rationale,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Decision':
        """
        Reconstruct Decision from dictionary.
        
        Inverse of to_dict(). Enables deserialization from storage/API.
        
        Parameters:
        ----------
        data : Dict[str, Any]
            Dictionary with keys: decision_type, rationale, metadata
            
        Returns:
        -------
        Decision object
        
        Raises:
        ------
        ValueError: If decision_type is not a valid DecisionType value
        KeyError: If required keys are missing
        """
        return cls(
            decision_type=DecisionType(data["decision_type"]),
            rationale=data["rationale"],
            metadata=data.get("metadata", {})
        )
    
    def __str__(self) -> str:
        """
        Human-readable string representation.
        
        Useful for logging and debugging.
        """
        return f"Decision({self.decision_type.value.upper()}: {self.rationale})"
    
    def __repr__(self) -> str:
        """
        Developer-friendly representation showing all fields.
        """
        return (
            f"Decision(decision_type={self.decision_type!r}, "
            f"rationale={self.rationale!r}, "
            f"metadata={self.metadata!r})"
        )


# -----------------------------------------------------------------------------
# Module-level helper functions
# -----------------------------------------------------------------------------

def create_pass_decision(
    rationale: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Decision:
    """
    Convenience factory for creating PASS decisions.
    
    Parameters:
    ----------
    rationale : str
        Explanation of why candidate passes
    metadata : Optional[Dict[str, Any]]
        Supporting data (default: empty dict)
        
    Returns:
    -------
    Decision with PASS type
    """
    return Decision(
        decision_type=DecisionType.PASS,
        rationale=rationale,
        metadata=metadata or {}
    )


def create_flag_decision(
    rationale: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Decision:
    """
    Convenience factory for creating FLAG decisions.
    
    Parameters:
    ----------
    rationale : str
        Explanation of what requires review
    metadata : Optional[Dict[str, Any]]
        Supporting data (default: empty dict)
        
    Returns:
    -------
    Decision with FLAG type
    """
    return Decision(
        decision_type=DecisionType.FLAG,
        rationale=rationale,
        metadata=metadata or {}
    )


def create_discard_decision(
    rationale: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Decision:
    """
    Convenience factory for creating DISCARD decisions.
    
    Parameters:
    ----------
    rationale : str
        Explanation of why candidate is discarded
    metadata : Optional[Dict[str, Any]]
        Supporting data (default: empty dict)
        
    Returns:
    -------
    Decision with DISCARD type
    """
    return Decision(
        decision_type=DecisionType.DISCARD,
        rationale=rationale,
        metadata=metadata or {}
    )


# -----------------------------------------------------------------------------
# Design Notes
# -----------------------------------------------------------------------------
# 
# Why separate Decision from decision-making logic?
# ------------------------------------------------
# 1. Single Responsibility: Decision is a data container, not business logic
# 2. Testability: Can test decision rendering/storage independently of logic
# 3. Flexibility: Can change decision criteria without changing Decision structure
# 4. Auditability: Decisions are self-documenting artifacts
# 
# Why use dataclass instead of plain dict?
# ----------------------------------------
# 1. Type safety: Ensures all decisions have required fields
# 2. IDE support: Auto-completion and type checking
# 3. Validation: __post_init__ enforces invariants
# 4. Immutability: Can use frozen=True if needed (currently mutable for flexibility)
# 
# Why include metadata as free-form dict?
# ---------------------------------------
# 1. Extensibility: Different agents may need different supporting data
# 2. Evolution: Can add new fields without breaking existing code
# 3. Integration: Easy to serialize/deserialize for storage
# 
# However, for production systems, consider:
# - Defining a metadata schema/protocol
# - Using TypedDict for common metadata patterns
# - Validating metadata structure in specialized Decision subclasses
#
