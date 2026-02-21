"""
agent/triage_agent.py

TriageAgent: Central orchestration controller for molecular triage workflow.

Week 5 Implementation — Similarity / IP-Risk Screening Integration

This module implements the deterministic, state-centric agent that coordinates
tool execution, maintains provenance, and delegates decision-making.

Design Principles:
    - Tools execute sequentially in fixed order (no parallelism)
    - Tools are stateless; AgentState is the single source of truth
    - Tool failures are non-terminal by default; execution continues with
      recorded errors
    - State becomes immutable after decision is set
    - Early termination allowed via explicit state signals
      (Week 3: invalid molecules; Week 4: SA > 7)
    - SimilarityTool never causes early termination — FLAG-and-continue
    - PolicyEngine evaluates completed state; does not control execution flow

Week 5 change summary:
    - Added SimilarityTool detection block after tool execution
    - FLAG result: annotate state.add_flag(reason=..., source="SimilarityTool")
    - ERROR result: annotate state.add_flag(reason=..., source="SimilarityTool")
    - No break — pipeline always continues after SimilarityTool
"""

from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone

import logging
from agent.agent_state import AgentState
from agent.decision import Decision, DecisionType
from policies.policy_engine import PolicyEngine
from tools.base_tool import Tool


class ToolExecutionStatus(Enum):
    """Status codes for tool execution outcomes."""
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class ToolFailureMode(Enum):
    """
    Tool failure handling policy.

    NON_TERMINAL: Failure is recorded; execution continues (default)
    TERMINAL: Failure aborts execution immediately
    """
    NON_TERMINAL = "non_terminal"
    TERMINAL = "terminal"


@dataclass
class ToolResult:
    """
    Encapsulates the output of a single tool execution.

    Attributes:
        tool_name: Identifier of the executed tool
        status: Execution outcome status
        data: Tool-specific output artifacts (structure defined by tool)
        error_message: Optional error description if status is FAILURE
        execution_time_ms: Time taken to execute tool in milliseconds
    """
    tool_name: str
    status: ToolExecutionStatus
    data: Optional[dict] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None


class ToolExecutionError(Exception):
    """Raised when a tool encounters an execution error."""
    pass


class StateValidationError(Exception):
    """Raised when AgentState validation fails."""
    pass


class PolicyEvaluationError(Exception):
    """Raised when PolicyEngine evaluation fails critically."""
    pass


class StateMutationError(Exception):
    """Raised when attempting to mutate finalized state."""
    pass


class TriageAgent:
    """
    Central orchestration controller for molecular triage workflow.

    Coordinates tool execution, maintains state, and delegates decision-making
    to the policy engine. Ensures deterministic, auditable execution with
    complete provenance tracking.

    Tool Execution Semantics (Week 5):
        1. ValidityTool  — runs first; invalid molecule → terminate early
        2. SAScoreTool   — runs second; SA > 7 → terminate early
        3. SimilarityTool — runs third; FLAG-only, never terminates early
        4. (future tools)

    Early termination signals:
        - ValidityTool: is_valid=False → state.terminate()
        - SAScoreTool: sa_decision="DISCARD" → state.terminate()
        - SimilarityTool: FLAG or ERROR → state.add_flag() only (no terminate)

    Attributes:
        tools: Ordered sequence of Tool instances to execute
        policy_engine: PolicyEngine instance for decision evaluation
        logger: Logging interface for execution tracking
    """

    def __init__(
        self,
        tools: List['Tool'],
        policy_engine: 'PolicyEngine',
        logger,
    ):
        """
        Initialize the triage agent with required dependencies.

        Args:
            tools: Ordered list of Tool instances. Execution order is list order.
            policy_engine: PolicyEngine instance for state evaluation
            logger: Logger instance for execution and provenance tracking

        Raises:
            ValueError: If tools list contains duplicate tool names
        """
        if not tools:
            logger.warning(
                "TriageAgent initialized with 0 tools — architectural validation mode"
            )

        tool_names = [tool.name for tool in tools]
        if len(tool_names) != len(set(tool_names)):
            duplicates = [n for n in tool_names if tool_names.count(n) > 1]
            raise ValueError(f"Duplicate tool names found: {set(duplicates)}")

        self.tools = tools
        self.policy_engine = policy_engine
        self.logger = logger

    def run(self, molecule_id: str, raw_input) -> 'AgentState':
        """
        Execute the complete triage workflow for a single molecule.

        Execution Flow:
            1. Initialize AgentState with molecule_id and raw_input
            2. Execute tools sequentially in fixed order
            3. Record all tool results (success or failure) in state
            4. Check for early termination between tools (validity, SA score)
            5. SimilarityTool: FLAG → add_flag(); ERROR → add_flag(); no break
            6. Delegate decision-making to PolicyEngine
            7. Finalize and return immutable state

        Args:
            molecule_id: Unique identifier for the molecule being triaged
            raw_input: Unprocessed input data (string SMILES, dict, or Molecule)

        Returns:
            AgentState: Complete final state with all tool results, metadata,
                       and final decision. State is immutable after this returns.

        Raises:
            StateValidationError: If state becomes invalid during execution
            PolicyEvaluationError: If policy evaluation fails critically
            ToolExecutionError: If a tool with TERMINAL failure mode fails
        """
        state = AgentState(
            molecule_id=molecule_id,
            raw_input=raw_input,
            execution_start_time=datetime.now(timezone.utc),
        )

        self.logger.info(
            "Starting triage run for molecule_id=%s", molecule_id,
            extra={"molecule_id": molecule_id},
        )

        # ── Sequential tool execution ────────────────────────────────────────
        for tool in self.tools:

            # Check for early termination signal from a previous tool
            if state.is_terminated():
                self.logger.info(
                    "Early termination requested: %s",
                    state.get_termination_reason(),
                    extra={
                        "molecule_id": molecule_id,
                        "termination_reason": state.get_termination_reason(),
                        "tools_completed": len(state.tool_results),
                    },
                )
                break

            tool_name = tool.name
            tool_start_time = self._get_timestamp()

            self.logger.debug(
                "Executing tool: %s",
                tool_name,
                extra={"molecule_id": molecule_id, "tool": tool_name},
            )

            try:
                result = tool.run(state)

                execution_time_ms = self._compute_elapsed_time_ms(tool_start_time)
                self._validate_tool_result(result, tool_name)

                if isinstance(result, ToolResult) and result.execution_time_ms is None:
                    result.execution_time_ms = execution_time_ms

                state.add_tool_result(tool_name=tool_name, result=result)

                # ── Post-run handling: ValidityTool (Week 3) ─────────────────
                if tool_name == "ValidityTool":
                    validity_data = (
                        result.data if isinstance(result, ToolResult) else result
                    )
                    is_valid = validity_data.get("is_valid", False)

                    if not is_valid:
                        error_msg = validity_data.get(
                            "error_message", "Unknown validation error"
                        )
                        state.add_message(
                            f"Molecule failed validity check: {error_msg}"
                        )
                        state.terminate(reason=f"Invalid chemistry: {error_msg}")
                        self.logger.warning(
                            "Terminating early for %s: molecule is chemically invalid",
                            molecule_id,
                            extra={
                                "molecule_id": molecule_id,
                                "validation_error": error_msg,
                                "tools_completed": len(state.tool_results),
                            },
                        )
                        break
                    else:
                        canonical_smiles = validity_data.get("smiles_canonical", "N/A")
                        num_atoms = validity_data.get("num_atoms", 0)
                        self.logger.info(
                            "Molecule %s passed validity check",
                            molecule_id,
                            extra={
                                "molecule_id": molecule_id,
                                "canonical_smiles": canonical_smiles,
                                "num_atoms": num_atoms,
                            },
                        )

                # ── Post-run handling: SAScoreTool (Week 4) ──────────────────
                elif tool_name == "SAScoreTool":
                    sa_data = (
                        result.data if isinstance(result, ToolResult) else result
                    )
                    sa_decision = sa_data.get("sa_decision", "PASS")
                    sa_score = sa_data.get("sa_score")

                    if sa_decision == "DISCARD":
                        error_msg = (
                            f"SA score {sa_score:.2f} exceeds discard threshold (> 7.0)"
                            if sa_score is not None
                            else "SA score DISCARD threshold exceeded"
                        )
                        state.add_message(f"Molecule failed SA check: {error_msg}")
                        state.terminate(reason=f"SA score too high: {error_msg}")
                        self.logger.warning(
                            "Terminating early for %s: %s",
                            molecule_id,
                            error_msg,
                            extra={
                                "molecule_id": molecule_id,
                                "sa_score": sa_score,
                                "sa_decision": sa_decision,
                                "tools_completed": len(state.tool_results),
                            },
                        )
                        break

                    elif sa_decision == "FLAG":
                        self.logger.info(
                            "Molecule %s SA score flagged: %.2f (6.0-7.0 range) — continuing",
                            molecule_id,
                            sa_score or 0.0,
                            extra={
                                "molecule_id": molecule_id,
                                "sa_score": sa_score,
                            },
                        )
                        # Annotate state — pipeline continues
                        if hasattr(state, "add_flag"):
                            state.add_flag(
                                reason=f"SA score {sa_score:.2f} is in flag range (6.0-7.0) — synthesis is challenging",
                                source="SAScoreTool",
                            )

                # ═══════════════════════════════════════════════════════════════
                # WEEK 5 ADDITION: Post-run handling for SimilarityTool
                # ═══════════════════════════════════════════════════════════════
                elif tool_name == "SimilarityTool":
                    sim_data = (
                        result.data if isinstance(result, ToolResult) else result
                    )
                    sim_decision = sim_data.get("similarity_decision", "PASS")
                    nn_tanimoto = sim_data.get("nearest_neighbor_tanimoto", 0.0)
                    nn_source = sim_data.get("nearest_neighbor_source")
                    nn_id = sim_data.get("nearest_neighbor_id")

                    if sim_decision == "FLAG":
                        self.logger.info(
                            "Molecule %s similarity FLAG: Tanimoto=%.3f from %s [%s] "
                            "— pipeline continues (no early termination)",
                            molecule_id,
                            nn_tanimoto,
                            nn_source or "N/A",
                            nn_id or "N/A",
                            extra={
                                "molecule_id": molecule_id,
                                "nearest_neighbor_tanimoto": nn_tanimoto,
                                "nearest_neighbor_source": nn_source,
                                "nearest_neighbor_id": nn_id,
                            },
                        )
                        # Annotate state — pipeline continues (no terminate())
                        if hasattr(state, "add_flag"):
                            state.add_flag(
                                reason=(
                                    f"Tanimoto {nn_tanimoto:.3f} >= "
                                    f"{sim_data.get('flag_threshold_used', 0.85):.2f} "
                                    f"threshold — nearest neighbor {nn_id} [{nn_source}], "
                                    f"IP review required"
                                ),
                                source="SimilarityTool",
                            )

                    elif sim_decision == "ERROR":
                        error_reason = sim_data.get("error_reason", "API unavailable")
                        self.logger.warning(
                            "Molecule %s similarity ERROR: %s — flagging conservatively, "
                            "pipeline continues",
                            molecule_id,
                            error_reason,
                            extra={
                                "molecule_id": molecule_id,
                                "similarity_error": error_reason,
                            },
                        )
                        # Annotate state — pipeline continues (no terminate())
                        if hasattr(state, "add_flag"):
                            state.add_flag(
                                reason=(
                                    f"Similarity APIs unavailable ({error_reason}) — "
                                    f"IP risk unknown, conservative flag applied"
                                ),
                                source="SimilarityTool",
                            )

                    else:
                        # similarity_decision == "PASS"
                        self.logger.info(
                            "Molecule %s similarity PASS: Tanimoto=%.3f (below threshold)",
                            molecule_id,
                            nn_tanimoto,
                            extra={
                                "molecule_id": molecule_id,
                                "nearest_neighbor_tanimoto": nn_tanimoto,
                            },
                        )
                # ═══════════════════════════════════════════════════════════════
                # END WEEK 5 ADDITION
                # ═══════════════════════════════════════════════════════════════

                self.logger.info(
                    "Tool %s completed successfully",
                    tool_name,
                    extra={
                        "molecule_id": molecule_id,
                        "tool": tool_name,
                        "status": ToolExecutionStatus.SUCCESS.value,
                        "execution_time_ms": execution_time_ms,
                    },
                )

            except ToolExecutionError as e:
                execution_time_ms = self._compute_elapsed_time_ms(tool_start_time)
                error_result = ToolResult(
                    tool_name=tool_name,
                    status=ToolExecutionStatus.FAILURE,
                    error_message=str(e),
                    execution_time_ms=execution_time_ms,
                )
                state.add_tool_result(tool_name=tool_name, result=error_result)
                self.logger.error(
                    "Tool %s failed: %s",
                    tool_name,
                    str(e),
                    extra={
                        "molecule_id": molecule_id,
                        "tool": tool_name,
                        "status": ToolExecutionStatus.FAILURE.value,
                        "error": str(e),
                        "execution_time_ms": execution_time_ms,
                    },
                )
                failure_mode = getattr(tool, "failure_mode", ToolFailureMode.NON_TERMINAL)
                if failure_mode == ToolFailureMode.TERMINAL:
                    self.logger.critical(
                        "Tool %s has TERMINAL failure mode — aborting",
                        tool_name,
                        extra={"molecule_id": molecule_id, "tool": tool_name},
                    )
                    raise

            except Exception as e:
                execution_time_ms = self._compute_elapsed_time_ms(tool_start_time)
                self.logger.critical(
                    "Unexpected error in tool %s: %s",
                    tool_name,
                    str(e),
                    extra={
                        "molecule_id": molecule_id,
                        "tool": tool_name,
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "execution_time_ms": execution_time_ms,
                    },
                )
                error_result = ToolResult(
                    tool_name=tool_name,
                    status=ToolExecutionStatus.FAILURE,
                    error_message=f"Unexpected error: {type(e).__name__}: {str(e)}",
                    execution_time_ms=execution_time_ms,
                )
                state.add_tool_result(tool_name=tool_name, result=error_result)
                raise

        # ── Tool execution phase complete ────────────────────────────────────
        state.add_message(f"All tools completed at {self._get_timestamp()}")

        self.logger.info(
            "Tool execution phase complete for molecule_id=%s",
            molecule_id,
            extra={
                "molecule_id": molecule_id,
                "total_tools": len(self.tools),
                "executed_tools": len(state.tool_results),
                "early_termination": state.is_terminated(),
            },
        )

        # ── PolicyEngine evaluation ──────────────────────────────────────────
        try:
            self.logger.debug(
                "Invoking PolicyEngine for molecule_id=%s", molecule_id,
                extra={"molecule_id": molecule_id},
            )
            decision = self.policy_engine.evaluate(state)
            self._validate_decision(decision)
            self.logger.info(
                "Policy evaluation complete: %s",
                decision.decision_type,
                extra={
                    "molecule_id": molecule_id,
                    "decision_type": str(decision.decision_type),
                },
            )

        except PolicyEvaluationError as e:
            self.logger.error(
                "Policy evaluation failed: %s",
                str(e),
                extra={"molecule_id": molecule_id, "error": str(e)},
            )
            raise

        except Exception as e:
            self.logger.critical(
                "Unexpected error during policy evaluation: %s",
                str(e),
                extra={
                    "molecule_id": molecule_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise PolicyEvaluationError(
                f"Unexpected policy evaluation error: {type(e).__name__}: {str(e)}"
            ) from e

        # ── Finalize state ───────────────────────────────────────────────────
        state.set_decision(decision)

        total_execution_time_ms = self._compute_total_execution_time(state)

        self.logger.info(
            "Triage run complete for molecule_id=%s",
            molecule_id,
            extra={
                "molecule_id": molecule_id,
                "decision_type": str(decision.decision_type),
                "total_tools": len(self.tools),
                "executed_tools": len(state.tool_results),
                "failed_tools": self._count_failed_tools(state),
                "total_execution_time_ms": total_execution_time_ms,
                "early_termination": state.is_terminated(),
            },
        )

        return state

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _get_timestamp(self) -> str:
        """Generate ISO-8601 UTC timestamp for provenance tracking."""
        return datetime.now(timezone.utc).isoformat()

    def _compute_elapsed_time_ms(self, start_timestamp: str) -> float:
        """Calculate elapsed time in milliseconds from an ISO start timestamp."""
        start_dt = datetime.fromisoformat(start_timestamp)
        end_dt = datetime.now(timezone.utc)
        return round((end_dt - start_dt).total_seconds() * 1000, 2)

    def _compute_total_execution_time(self, state: AgentState) -> float:
        """Compute total wall-clock execution time in milliseconds."""
        start_time = state.execution_start_time
        end_time = state.get_decision_timestamp()

        if start_time is None or end_time is None:
            return 0.0

        start_dt = (
            datetime.fromisoformat(start_time)
            if isinstance(start_time, str) else start_time
        )
        end_dt = (
            datetime.fromisoformat(end_time)
            if isinstance(end_time, str) else end_time
        )

        return (end_dt - start_dt).total_seconds() * 1000

    def _validate_tool_result(self, result, tool_name: str) -> None:
        """Raise StateValidationError if tool returned None."""
        if result is None:
            raise StateValidationError(
                f"Tool {tool_name} returned None; expected ToolResult or dict"
            )

    def _validate_decision(self, decision) -> None:
        """Raise StateValidationError if decision is malformed."""
        if decision is None:
            raise StateValidationError("PolicyEngine returned None decision")
        if not hasattr(decision, "decision_type"):
            raise StateValidationError("Decision missing required 'decision_type' field")

    def _count_failed_tools(self, state: AgentState) -> int:
        """Count tools with FAILURE status in executed results."""
        failed = 0
        for result in state.tool_results.values():
            if (
                isinstance(result, ToolResult)
                and result.status == ToolExecutionStatus.FAILURE
            ):
                failed += 1
        return failed