"""
agent/agent_state.py

Central data structure for a single agent evaluation run.

Week 2-3: Core state, tool results, termination support, execution timing.
Week 4:   Added add_flag(), get_flags(), is_flagged() for FLAG annotation.
          FLAGS are non-terminating — molecule continues through pipeline
          with an audit trail of concerns.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


@dataclass
class AgentState:
    """
    Container for all intermediate and final information
    accumulated during a single agent run.

    Attributes:
        molecule_id:          Unique identifier for the molecule being evaluated.
        raw_input:            Original input data (SMILES string, dict, or Molecule).
        tool_results:         Dict mapping tool name -> tool output.
        messages:             Chronological log of agent actions.
        decision:             Final Decision object (None until set at end of run).
        execution_start_time: UTC datetime when the run began.
        _tools_complete:      Internal flag: all tools have finished.
        _terminated:          Internal flag: early termination was signalled.
        _decision_set:        Internal flag: final decision has been recorded.
    """

    # Required fields
    molecule_id: str
    raw_input: Any

    # Optional fields with defaults
    tool_results: Dict[str, Any] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    decision: Optional[Any] = None
    execution_start_time: Optional[datetime] = None

    # Internal state flags (repr=False keeps them out of default __repr__)
    _tools_complete: bool = field(default=False, repr=False)
    _terminated: bool = field(default=False, repr=False)
    _decision_set: bool = field(default=False, repr=False)

    # Note: _flags is NOT a dataclass field — it is initialised lazily in
    # add_flag() using object.__setattr__ to avoid changing __init__ signature.

    # =========================================================================
    # Core Methods (Week 2-3, unchanged)
    # =========================================================================

    def add_tool_result(self, tool_name: str, result: Any) -> None:
        """
        Store output produced by a tool.

        Args:
            tool_name: Unique name of the tool (e.g. "ValidityTool").
            result:    Tool output — may be a plain dict or ToolResult dataclass.
        """
        self.tool_results[tool_name] = result

    def add_message(self, message: str) -> None:
        """Append a human-readable message for traceability."""
        self.messages.append(message)

    def set_decision(self, decision: Any) -> None:
        """Attach the final decision object (called once at end of run)."""
        self.decision = decision
        self._decision_set = True

    def get_decision_timestamp(self) -> Optional[datetime]:
        """
        Get the timestamp when the decision was set.

        Reads from decision.metadata['timestamp'] if available,
        otherwise returns current UTC time.
        """
        if self.decision is None:
            return None

        if hasattr(self.decision, 'metadata') and isinstance(self.decision.metadata, dict):
            timestamp = self.decision.metadata.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        return dt
                    except Exception:
                        pass
                elif isinstance(timestamp, datetime):
                    if timestamp.tzinfo is None:
                        return timestamp.replace(tzinfo=timezone.utc)
                    return timestamp

        return datetime.now(timezone.utc)

    @property
    def identifier(self) -> str:
        """
        Alias for molecule_id, used by PolicyEngine for validation/logging.
        Keeps AgentState compatible with the Policy interface contract.
        """
        return self.molecule_id

    # =========================================================================
    # State Management Methods (Week 2-3, unchanged)
    # =========================================================================

    def set_tools_complete(self, timestamp: Optional[str] = None) -> None:
        """Mark that all tools have completed execution."""
        self._tools_complete = True
        if timestamp:
            self.add_message(f"All tools completed at {timestamp}")

    def is_tools_complete(self) -> bool:
        """Return True if all tools have completed execution."""
        return self._tools_complete

    def terminate(self, reason: Optional[str] = None) -> None:
        """
        Signal early termination of agent execution.

        After calling this, triage_agent.py will stop the tool loop
        before running the next tool.

        Args:
            reason: Human-readable explanation for termination.
        """
        self._terminated = True
        if reason:
            self.add_message(f"Early termination: {reason}")

    def is_terminated(self) -> bool:
        """Return True if early termination has been signalled."""
        return self._terminated

    def get_termination_reason(self) -> Optional[str]:
        """
        Return the most recent termination message, or None if not terminated.
        """
        if not self._terminated:
            return None
        for msg in reversed(self.messages):
            if "termination" in msg.lower():
                return msg
        return "Early termination (reason not specified)"

    def is_decision_set(self) -> bool:
        """Return True if a final decision has been recorded."""
        return self._decision_set

    def get_tool_result(self, tool_name: str) -> Optional[Any]:
        """Return the stored result for a named tool, or None."""
        return self.tool_results.get(tool_name)

    def has_tool_result(self, tool_name: str) -> bool:
        """Return True if a result exists for the named tool."""
        return tool_name in self.tool_results

    def get_all_messages(self) -> List[str]:
        """Return a copy of all messages in chronological order."""
        return self.messages.copy()

    # =========================================================================
    # FLAG Annotation Methods (Week 4)
    # =========================================================================

    def add_flag(self, reason: str, source: str) -> None:
        """
        Record a FLAG annotation without terminating the pipeline.

        Unlike terminate(), this does NOT stop tool execution. The molecule
        continues through the pipeline with a warning recorded. Multiple flags
        from different tools accumulate.

        Args:
            reason: Human-readable explanation of the concern.
            source: Name of the tool or component raising the flag
                    (e.g. "SAScoreTool").
        """
        # Initialise lazily via object.__setattr__ to avoid touching the
        # dataclass __init__ signature. Works for non-frozen dataclasses.
        if not hasattr(self, '_flags') or self._flags is None:
            object.__setattr__(self, '_flags', [])
        self._flags.append({
            'reason': reason,
            'source': source,
        })
        self.add_message(f"FLAG from {source}: {reason}")

    def get_flags(self) -> List[Dict[str, str]]:
        """
        Return all FLAG annotations accumulated during this run.

        Returns:
            List of dicts, each with 'reason' and 'source' keys.
            Empty list if no flags have been raised.
        """
        return list(getattr(self, '_flags', []))

    def is_flagged(self) -> bool:
        """Return True if any FLAG annotations exist on this state."""
        return len(self.get_flags()) > 0

    # =========================================================================
    # Serialisation
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialisation/logging."""
        return {
            "molecule_id": self.molecule_id,
            "raw_input": self.raw_input,
            "tool_results": self.tool_results,
            "messages": self.messages,
            "decision": self.decision,
            "execution_start_time": (
                self.execution_start_time.isoformat()
                if self.execution_start_time else None
            ),
            "tools_complete": self._tools_complete,
            "terminated": self._terminated,
            "decision_set": self._decision_set,
            "flags": self.get_flags(),
        }