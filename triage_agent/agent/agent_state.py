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

# Import dataclass for clean, structured class definition with less boilerplate
from dataclasses import dataclass, field
# Import type hints for better code documentation and IDE support
from typing import Any, Dict, List, Optional


# Using @dataclass decorator automatically generates __init__, __repr__, __eq__, etc.
# This reduces boilerplate code while maintaining clarity
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
    """

    # Required fields - must be provided when creating an AgentState instance
    # These identify what molecule we're working with and what data we started with
    molecule_id: str  # Unique identifier for tracking and logging
    raw_input: Any    # Original input data - kept for auditability

    # Optional fields with default factories - automatically initialized as empty
    # Using field(default_factory=dict) instead of {} prevents mutable default argument issues
    # All instances get their own separate dict/list, not a shared reference
    
    tool_results: Dict[str, Any] = field(default_factory=dict)
    # Stores outputs from each tool that runs
    # Example: {"toxicity_predictor": {"score": 0.23}, "solubility_check": {"soluble": True}}
    
    messages: List[str] = field(default_factory=list)
    # Chronological log of what happened during the run
    # Example: ["Started evaluation", "Ran toxicity check", "Decision: Accept"]

    decision: Optional[Any] = None
    # Final outcome - starts as None, gets set exactly once at the end
    # Optional[Any] means it can be None or any type
    # Example: {"accept": True, "confidence": 0.87, "reason": "Low toxicity"}

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
        # Simple dictionary assignment - store the result under the tool's name
        # This creates or updates the entry for this tool
        self.tool_results[tool_name] = result

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
        # Append to the list - maintains chronological order
        # Each message gets added to the end of the list
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
        # Direct assignment - replace whatever was there before (usually None)
        # This marks the transition from "evaluating" to "decided"
        self.decision = decision
