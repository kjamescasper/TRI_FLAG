"""agent/agent_state.py

This module defines the AgentState class, which serves as the central data structure
for storing all information accumulated during a single agent evaluation run.

The AgentState acts as shared memory across tools, policies, and decision logic,
enabling transparency, reproducibility, and explainability of agent behavior.

Design Philosophy:
    - Explicit state enables explainability (why a decision happened)
    - Modularity (tools don't depend on each other, only on state)
    - Reproducibility (state can be serialized and replayed)
    - Future LLM integration (LLMs can read and summarize state)

Constraints:
    - Must NOT perform computation (data container only)
    - Must NOT apply policy logic (leave that to policy modules)
    - Must NOT contain domain-specific assumptions (stay generic)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

@dataclass
class AgentState:
    """
    Container for all intermediate and final information
    accumulated during a single agent run.

    This class stores:
    - Input data (molecule identifier and raw input)
    - Tool execution results
    - Human-readable messages for traceability
    - Final decision object
    - Execution metadata and state flags

    The state is designed to be mutable and incrementally updated as the
    agent progresses through its evaluation workflow.

    Attributes:
        molecule_id: Unique identifier for the molecule being evaluated
                     (e.g., "MOL_12345" or a SMILES string identifier)
        raw_input: Original input data provided to the agent
                   Format depends on input source (could be dict, string, object, etc.)
                   Preserved for reproducibility and debugging
        tool_results: Dictionary mapping tool names to their output results
                      Key: tool name (str), Value: tool output (Any type)
                      Allows tools to read each other's results without direct coupling
        messages: List of human-readable messages for logging and debugging
                  Provides a chronological narrative of what the agent did
        decision: Final decision object produced by the agent (None until set)
                  Remains None during evaluation, set once at the end
        execution_start_time: Timestamp when execution began (for performance tracking)
        _tools_complete: Internal flag indicating all tools have finished execution
        _terminated: Internal flag for early termination signal
        _decision_set: Internal flag indicating decision has been finalized
    """

    # Required fields - must be provided when creating an AgentState instance
    molecule_id: str  # Unique identifier for tracking and logging
    raw_input: Any    # Original input data - kept for auditability

    # Optional fields with default factories - automatically initialized as empty
    tool_results: Dict[str, Any] = field(default_factory=dict)
    # Stores outputs from each tool that runs
    # Example: {"toxicity_predictor": {"score": 0.23}, "solubility_check": {"soluble": True}}
    
    messages: List[str] = field(default_factory=list)
    # Chronological log of what happened during the run
    # Example: ["Started evaluation", "Ran toxicity check", "Decision: Accept"]

    decision: Optional[Any] = None
    # Final outcome - starts as None, gets set exactly once at the end
    
    # Time tracking (Week 3 addition)
    execution_start_time: Optional[datetime] = None

    # Internal state tracking flags (Week 2-3 additions for TriageAgent)
    # These are prefixed with _ to indicate they're internal/private
    _tools_complete: bool = field(default=False, repr=False)
    _terminated: bool = field(default=False, repr=False)
    _decision_set: bool = field(default=False, repr=False)

    # =========================================================================
    # Core Methods (Original Design)
    # =========================================================================

    def add_tool_result(self, tool_name: str, result: Any) -> None:
        """
        Store output produced by a tool.

        This method updates the tool_results dictionary with the output
        from a specific tool execution. This enables:
        - Tools to be decoupled (they don't call each other directly)
        - Later tools to access earlier results via the shared state
        - Complete audit trail of what each tool produced

        Args:
            tool_name: Name of the tool that produced the result
                      Should be unique and descriptive (e.g., "toxicity_predictor")
            result: Output data from the tool
                   Format varies by tool - could be dict, number, object, etc.
                   No type restrictions to maintain flexibility

        Returns:
            None - modifies state in place

        Example:
            >>> state = AgentState(molecule_id="MOL_001", raw_input={})
            >>> state.add_tool_result("toxicity_predictor", {"score": 0.23})
            >>> state.tool_results
            {"toxicity_predictor": {"score": 0.23}}

        Note:
            If the same tool_name is used twice, the second call overwrites
            the first result. This is intentional - tools shouldn't run twice.
        """
        self.tool_results[tool_name] = result

    def get_decision_timestamp(self) -> Optional[datetime]:
        """
        Get the timestamp when the decision was set.
    
        Returns:
           Timezone-aware datetime when decision was set, or None
        """
        if self.decision is None:
            return None
    
        # Try to get timestamp from decision metadata
        if hasattr(self.decision, 'metadata') and isinstance(self.decision.metadata, dict):
            timestamp = self.decision.metadata.get('timestamp')
            if timestamp:
                # If it's a string (ISO format), parse it to datetime
                if isinstance(timestamp, str):
                    try:
                        # Parse ISO format and ensure timezone awareness
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        # If no timezone, add UTC
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        return dt
                    except:
                        pass
                # If it's already a datetime, return it
                elif isinstance(timestamp, datetime):
                    # Ensure timezone awareness
                    if timestamp.tzinfo is None:
                        return timestamp.replace(tzinfo=timezone.utc)
                    return timestamp
    
        # Fallback: return current time with timezone
        return datetime.now(timezone.utc)

    def add_message(self, message: str) -> None:
        """
        Append a human-readable message for traceability.

        Messages provide a chronological log of agent actions and decisions,
        useful for:
        - Debugging when something goes wrong
        - Explainability (showing users why a decision was made)
        - Audit trails for regulatory compliance
        - Understanding agent behavior during development

        Args:
            message: Human-readable description of an event or action
                    Should be clear and informative
                    Examples: "Starting toxicity evaluation"
                             "Toxicity score below threshold"
                             "Decision: Accept molecule"

        Returns:
            None - modifies state in place

        Example:
            >>> state = AgentState(molecule_id="MOL_001", raw_input={})
            >>> state.add_message("Starting toxicity evaluation")
            >>> state.add_message("Toxicity score: 0.23")
            >>> state.messages
            ["Starting toxicity evaluation", "Toxicity score: 0.23"]

        Note:
            Messages are appended in order, creating a timeline of events.
            This preserves the sequence of operations for later analysis.
        """
        self.messages.append(message)

    def set_decision(self, decision: Any) -> None:
        """
        Attach the final decision object.

        This should be called once at the end of the agent workflow to
        record the final outcome. This is the culmination of all tool
        results and policy evaluations.

        Design rationale:
        - Separate method (not direct assignment) for explicitness
        - Makes it clear when the final decision is being made
        - Could add validation or logging in future versions

        Args:
            decision: The final decision object
                     Format depends on the decision type and domain
                     Could be: dict, custom Decision class, boolean, etc.
                     Example formats:
                       {"accept": True, "confidence": 0.87}
                       Decision(action="accept", rationale="Low toxicity")

        Returns:
            None - modifies state in place

        Example:
            >>> state = AgentState(molecule_id="MOL_001", raw_input={})
            >>> state.set_decision({"accept": True, "confidence": 0.87})
            >>> state.decision
            {"accept": True, "confidence": 0.87}

        Warning:
            This should typically only be called once per agent run.
            Calling it multiple times will overwrite the previous decision.
            Consider adding validation in future versions if this becomes an issue.
        """
        self.decision = decision
        self._decision_set = True

    @property
    def identifier(self) -> str:
        """
        Alias for molecule_id to support PolicyEngine interface compatibility.
        
        The PolicyEngine expects state.identifier for validation and logging.
        This property provides that interface while keeping molecule_id as the
        canonical field name in AgentState.
        
        Returns:
            The molecule identifier (same as molecule_id)
        
        Example:
            >>> state = AgentState(molecule_id="MOL_001", raw_input={})
            >>> state.identifier
            "MOL_001"
            >>> state.identifier == state.molecule_id
            True
        """
        return self.molecule_id

    # =========================================================================
    # Week 2-3 Extensions: State Management Methods
    # These methods support TriageAgent's execution flow tracking
    # =========================================================================
    
    def set_tools_complete(self, timestamp: Optional[str] = None) -> None:
        """
        Mark that all tools have completed execution.
        
        This signals the transition from tool execution phase to policy
        evaluation phase. Optional timestamp can be provided for audit trail.
        
        Args:
            timestamp: Optional ISO format timestamp when tools completed
        
        Returns:
            None - modifies state in place
        
        Example:
            >>> state = AgentState(molecule_id="MOL_001", raw_input={})
            >>> state.set_tools_complete(timestamp="2024-02-09T15:30:00Z")
            >>> state.is_tools_complete()
            True
        """
        self._tools_complete = True
        if timestamp:
            self.add_message(f"All tools completed at {timestamp}")

    def is_tools_complete(self) -> bool:
        """
        Check if all tools have completed execution.
        
        Returns:
            True if tools are complete, False otherwise
        
        Example:
            >>> state = AgentState(molecule_id="MOL_001", raw_input={})
            >>> state.is_tools_complete()
            False
            >>> state.set_tools_complete()
            >>> state.is_tools_complete()
            True
        """
        return self._tools_complete

    def terminate(self, reason: Optional[str] = None) -> None:
        """
        Signal early termination of agent execution.
        
        Sets internal flag that can be checked by the agent to stop
        execution early. Optional reason can be provided for logging.
        
        Args:
            reason: Optional explanation for why execution is terminating
        
        Returns:
            None - modifies state in place
        
        Example:
            >>> state = AgentState(molecule_id="MOL_001", raw_input={})
            >>> state.terminate(reason="Critical validation failure")
            >>> state.is_terminated()
            True
        """
        self._terminated = True
        if reason:
            self.add_message(f"Early termination: {reason}")

    def is_terminated(self) -> bool:
        """
        Check if early termination has been signaled.
        
        Returns:
            True if terminated, False otherwise
        
        Example:
            >>> state = AgentState(molecule_id="MOL_001", raw_input={})
            >>> state.is_terminated()
            False
            >>> state.terminate()
            >>> state.is_terminated()
            True
        """
        return self._terminated

    def get_termination_reason(self) -> Optional[str]:
        """
        Get the reason for early termination if terminated.
        
        Searches messages in reverse chronological order for termination message.
        
        Returns:
            Termination reason from messages, or None if not terminated
        
        Example:
            >>> state = AgentState(molecule_id="MOL_001", raw_input={})
            >>> state.terminate(reason="Invalid chemistry")
            >>> state.get_termination_reason()
            "Early termination: Invalid chemistry"
        """
        if not self._terminated:
            return None
        
        # Look for termination message (search backwards for most recent)
        for msg in reversed(self.messages):
            if "termination" in msg.lower():
                return msg
        
        return "Early termination (reason not specified)"

    def is_decision_set(self) -> bool:
        """
        Check if a decision has been finalized.
        
        Returns:
            True if decision has been set, False otherwise
        
        Example:
            >>> state = AgentState(molecule_id="MOL_001", raw_input={})
            >>> state.is_decision_set()
            False
            >>> state.set_decision({"accept": True})
            >>> state.is_decision_set()
            True
        """
        return self._decision_set

    def get_tool_result(self, tool_name: str) -> Optional[Any]:
        """
        Retrieve the result from a specific tool.
        
        Args:
            tool_name: Name of the tool whose result to retrieve
        
        Returns:
            Tool result if found, None otherwise
        
        Example:
            >>> state = AgentState(molecule_id="MOL_001", raw_input={})
            >>> state.add_tool_result("toxicity", {"score": 0.23})
            >>> state.get_tool_result("toxicity")
            {"score": 0.23}
            >>> state.get_tool_result("nonexistent")
            None
        """
        return self.tool_results.get(tool_name)

    def has_tool_result(self, tool_name: str) -> bool:
        """
        Check if a specific tool has produced a result.
        
        Args:
            tool_name: Name of the tool to check
        
        Returns:
            True if tool result exists, False otherwise
        
        Example:
            >>> state = AgentState(molecule_id="MOL_001", raw_input={})
            >>> state.add_tool_result("toxicity", {"score": 0.23})
            >>> state.has_tool_result("toxicity")
            True
            >>> state.has_tool_result("nonexistent")
            False
        """
        return tool_name in self.tool_results

    def get_all_messages(self) -> List[str]:
        """
        Retrieve all messages in chronological order.
        
        Returns:
            List of all messages
        
        Example:
            >>> state = AgentState(molecule_id="MOL_001", raw_input={})
            >>> state.add_message("First message")
            >>> state.add_message("Second message")
            >>> state.get_all_messages()
            ["First message", "Second message"]
        """
        return self.messages.copy()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to dictionary for serialization.
        
        Useful for:
        - JSON serialization for API responses
        - Database storage
        - Logging and debugging
        
        Returns:
            Dictionary representation of the state
        
        Example:
            >>> state = AgentState(molecule_id="MOL_001", raw_input={"smiles": "CCO"})
            >>> state.add_message("Test message")
            >>> state_dict = state.to_dict()
            >>> state_dict["molecule_id"]
            "MOL_001"
        """
        return {
            "molecule_id": self.molecule_id,
            "raw_input": self.raw_input,
            "tool_results": self.tool_results,
            "messages": self.messages,
            "decision": self.decision,
            "execution_start_time": self.execution_start_time.isoformat() if self.execution_start_time else None,
            "tools_complete": self._tools_complete,
            "terminated": self._terminated,
            "decision_set": self._decision_set
        }


# =============================================================================
# Design Notes (Week 2-3)
# =============================================================================
#
# Changes from original AgentState:
# ---------------------------------
# 1. Added state tracking flags (_tools_complete, _terminated, _decision_set)
#    - These support TriageAgent's execution flow control
#    - Prefixed with _ to indicate internal use
#    - repr=False keeps them out of default string representation
#
# 2. Added query methods (is_terminated, is_tools_complete, etc.)
#    - Provides clean API for checking state without direct flag access
#    - Supports future extension (e.g., logging when flags are checked)
#
# 3. Added helper methods (get_tool_result, has_tool_result, etc.)
#    - Convenience methods that TriageAgent and PolicyEngine expect
#    - Maintains consistency with original design (explicit methods, not magic)
#
# 4. Added execution_start_time field (Week 3)
#    - Enables performance tracking and timing analysis
#    - Used to compute total execution time
#
# 5. Added get_termination_reason() method (Week 3)
#    - Allows agent to report WHY it terminated early
#    - Useful for debugging and user feedback
#
# 6. Updated set_decision to set the _decision_set flag
#    - Allows agent to detect when decision has been finalized
#    - Supports immutability enforcement in future versions
#
# Why these changes maintain the original design philosophy:
# ----------------------------------------------------------
# - Still a data container (no computation or policy logic)
# - Still generic (no domain-specific assumptions)
# - Still explicit (all state changes via clear method calls)
# - Still reproducible (all state is serializable via to_dict)
# - Still modular (tools and policies don't depend on each other)
#
# The additions are purely mechanical bookkeeping to support orchestration,
# not domain logic or decision-making.
#