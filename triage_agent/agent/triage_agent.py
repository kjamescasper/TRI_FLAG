"""
agent/triage_agent.py

TriageAgent: Central orchestration controller for molecular triage workflow.

Week 3: Chemical validity checking (ValidityTool early termination)
Week 4: SA score checking (SAScoreTool early termination on DISCARD,
        FLAG annotation on FLAG, continues on PASS)

All Week 3 code is preserved exactly. Week 4 adds one new elif block
after the ValidityTool block. No other changes.
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
        tool_name:        Identifier of the executed tool
        status:           Execution outcome status
        data:             Tool-specific output (structure defined by tool)
        error_message:    Optional error description if status is FAILURE
        execution_time_ms: Time taken in milliseconds
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
    """Raised when attempting to mutate finalised state."""
    pass


class TriageAgent:
    """
    Central orchestration controller for molecular triage workflow.

    Tool Execution Semantics:
        - Tools execute sequentially in the order provided at construction
        - ValidityTool runs first; invalid molecules terminate early (Week 3)
        - SAScoreTool runs second; SA > 7 terminates, 6-7 flags and
          continues, < 6 continues silently (Week 4)
        - Tool failures are non-terminal by default
        - PolicyEngine evaluates the final state; it does not control
          execution flow

    Attributes:
        tools:         Ordered sequence of Tool instances to execute
        policy_engine: PolicyEngine instance for decision evaluation
        logger:        Logging interface for execution tracking
    """

    def __init__(
        self,
        tools: List['Tool'],
        policy_engine: 'PolicyEngine',
        logger
    ):
        if not tools:
            logger.warning("TriageAgent initialised with 0 tools — architectural validation mode")

        tool_names = [tool.name for tool in tools]
        if len(tool_names) != len(set(tool_names)):
            duplicates = [name for name in tool_names if tool_names.count(name) > 1]
            raise ValueError(f"Duplicate tool names found: {set(duplicates)}")

        self.tools = tools
        self.policy_engine = policy_engine
        self.logger = logger

    def run(self, molecule_id: str, raw_input: dict) -> 'AgentState':
        """
        Execute the complete triage workflow for a single molecule.

        Execution Flow:
        1. Initialise AgentState with molecule_id and raw_input
        2. Execute tools sequentially in fixed order
        3. After each tool: check for early termination or FLAG signals
        4. Delegate final decision to PolicyEngine
        5. Finalise and return state

        Args:
            molecule_id: Unique identifier for the molecule being triaged
            raw_input:   SMILES string, dict with 'smiles' key, or Molecule object

        Returns:
            AgentState: Complete final state with all tool results and decision.
        """
        state = AgentState(
            molecule_id=molecule_id,
            raw_input=raw_input,
            execution_start_time=datetime.now(timezone.utc)
        )

        self.logger.info(
            f"Starting triage run for molecule_id={molecule_id}",
            extra={"molecule_id": molecule_id}
        )

        for tool in self.tools:
            # Check early termination signal before running next tool
            if state.is_terminated():
                self.logger.info(
                    f"Early termination requested: {state.get_termination_reason()}",
                    extra={
                        "molecule_id": molecule_id,
                        "termination_reason": state.get_termination_reason(),
                        "tools_completed": len(state.tool_results)
                    }
                )
                break

            tool_name = tool.name
            tool_start_time = self._get_timestamp()

            self.logger.debug(
                f"Executing tool: {tool_name}",
                extra={"molecule_id": molecule_id, "tool": tool_name}
            )

            try:
                result = tool.run(state)
                execution_time_ms = self._compute_elapsed_time_ms(tool_start_time)
                self._validate_tool_result(result, tool_name)

                if isinstance(result, ToolResult) and result.execution_time_ms is None:
                    result.execution_time_ms = execution_time_ms

                state.add_tool_result(tool_name=tool_name, result=result)

                # ═══════════════════════════════════════════════════════════
                # WEEK 3: ValidityTool early termination
                # ═══════════════════════════════════════════════════════════
                if tool_name == "ValidityTool":
                    # Unwrap ToolResult wrapper if present
                    if isinstance(result, ToolResult):
                        validity_data = result.data
                    else:
                        validity_data = result

                    is_valid = validity_data.get('is_valid', False)

                    if not is_valid:
                        error_msg = validity_data.get('error_message', 'Unknown validation error')

                        state.add_message(
                            f"Molecule failed validity check: {error_msg}"
                        )
                        state.terminate(reason=f"Invalid chemistry: {error_msg}")

                        self.logger.warning(
                            f"Terminating early for {molecule_id}: molecule is chemically invalid",
                            extra={
                                "molecule_id": molecule_id,
                                "validation_error": error_msg,
                                "tools_completed": len(state.tool_results)
                            }
                        )
                        break
                    else:
                        canonical_smiles = validity_data.get('smiles_canonical', 'N/A')
                        num_atoms = validity_data.get('num_atoms', 0)

                        self.logger.info(
                            f"Molecule {molecule_id} passed validity check",
                            extra={
                                "molecule_id": molecule_id,
                                "canonical_smiles": canonical_smiles,
                                "num_atoms": num_atoms
                            }
                        )

                # ═══════════════════════════════════════════════════════════
                # WEEK 4: SAScoreTool — terminate on DISCARD/ERROR,
                #         annotate on FLAG, continue silently on PASS
                # ═══════════════════════════════════════════════════════════
                elif tool_name == "SAScoreTool":
                    # Unwrap ToolResult wrapper if present (same pattern as Week 3)
                    if isinstance(result, ToolResult):
                        sa_data = result.data or {}
                    else:
                        sa_data = result

                    sa_decision = sa_data.get('sa_decision', 'PASS')

                    if sa_decision == 'DISCARD':
                        reason = sa_data.get(
                            'sa_description',
                            f"SA score {sa_data.get('sa_score', '?')} exceeds discard threshold"
                        )
                        state.terminate(reason=reason)
                        self.logger.warning(
                            f"Terminating early for {molecule_id}: SA score too high",
                            extra={
                                "molecule_id": molecule_id,
                                "sa_score": sa_data.get('sa_score'),
                                "category": sa_data.get('synthesizability_category'),
                                "reason": reason,
                                "tools_completed": len(state.tool_results)
                            }
                        )
                        break

                    elif sa_decision == 'FLAG':
                        # Annotate state and continue — do NOT break
                        reason = sa_data.get(
                            'sa_description',
                            f"SA score {sa_data.get('sa_score', '?')} in challenging range"
                        )
                        state.add_flag(reason=reason, source="SAScoreTool")
                        self.logger.warning(
                            f"FLAG for {molecule_id}: challenging SA score, pipeline continues",
                            extra={
                                "molecule_id": molecule_id,
                                "sa_score": sa_data.get('sa_score'),
                                "category": sa_data.get('synthesizability_category'),
                                "warning_flags": sa_data.get('warning_flags', [])
                            }
                        )
                        # No break — next tool runs normally

                    elif sa_decision == 'ERROR':
                        reason = sa_data.get('error_message', 'SA score computation failed')
                        state.terminate(reason=f"SAScoreTool error: {reason}")
                        self.logger.error(
                            f"Terminating early for {molecule_id}: SAScoreTool error",
                            extra={
                                "molecule_id": molecule_id,
                                "error": reason
                            }
                        )
                        break

                    else:
                        # PASS — log and continue silently
                        self.logger.info(
                            f"Molecule {molecule_id} passed SA score check",
                            extra={
                                "molecule_id": molecule_id,
                                "sa_score": sa_data.get('sa_score'),
                                "category": sa_data.get('synthesizability_category'),
                            }
                        )
                # ═══════════════════════════════════════════════════════════
                # END WEEK 4
                # ═══════════════════════════════════════════════════════════

                self.logger.info(
                    f"Tool {tool_name} completed successfully",
                    extra={
                        "molecule_id": molecule_id,
                        "tool": tool_name,
                        "status": ToolExecutionStatus.SUCCESS.value,
                        "execution_time_ms": execution_time_ms
                    }
                )

            except ToolExecutionError as e:
                execution_time_ms = self._compute_elapsed_time_ms(tool_start_time)
                error_result = ToolResult(
                    tool_name=tool_name,
                    status=ToolExecutionStatus.FAILURE,
                    error_message=str(e),
                    execution_time_ms=execution_time_ms
                )
                state.add_tool_result(tool_name=tool_name, result=error_result)
                self.logger.error(
                    f"Tool {tool_name} failed: {str(e)}",
                    extra={
                        "molecule_id": molecule_id,
                        "tool": tool_name,
                        "status": ToolExecutionStatus.FAILURE.value,
                        "error": str(e),
                        "execution_time_ms": execution_time_ms
                    }
                )
                failure_mode = getattr(tool, 'failure_mode', ToolFailureMode.NON_TERMINAL)
                if failure_mode == ToolFailureMode.TERMINAL:
                    self.logger.critical(
                        f"Tool {tool_name} has TERMINAL failure mode; aborting execution",
                        extra={"molecule_id": molecule_id, "tool": tool_name}
                    )
                    raise

            except Exception as e:
                execution_time_ms = self._compute_elapsed_time_ms(tool_start_time)
                self.logger.critical(
                    f"Unexpected error in tool {tool_name}: {str(e)}",
                    extra={
                        "molecule_id": molecule_id,
                        "tool": tool_name,
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "execution_time_ms": execution_time_ms
                    }
                )
                error_result = ToolResult(
                    tool_name=tool_name,
                    status=ToolExecutionStatus.FAILURE,
                    error_message=f"Unexpected error: {type(e).__name__}: {str(e)}",
                    execution_time_ms=execution_time_ms
                )
                state.add_tool_result(tool_name=tool_name, result=error_result)
                raise

        state.add_message(f"All tools completed at {self._get_timestamp()}")

        self.logger.info(
            f"Tool execution phase complete for molecule_id={molecule_id}",
            extra={
                "molecule_id": molecule_id,
                "total_tools": len(self.tools),
                "executed_tools": len(state.tool_results),
                "early_termination": state.is_terminated()
            }
        )

        # Delegate decision-making to PolicyEngine
        try:
            self.logger.debug(
                f"Invoking PolicyEngine for molecule_id={molecule_id}",
                extra={"molecule_id": molecule_id}
            )
            decision = self.policy_engine.evaluate(state)
            self._validate_decision(decision)
            self.logger.info(
                f"Policy evaluation complete: {decision.decision_type}",
                extra={
                    "molecule_id": molecule_id,
                    "decision_decision_type": decision.decision_type,
                }
            )

        except PolicyEvaluationError as e:
            self.logger.error(
                f"Policy evaluation failed: {str(e)}",
                extra={"molecule_id": molecule_id, "error": str(e)}
            )
            raise

        except Exception as e:
            self.logger.critical(
                f"Unexpected error during policy evaluation: {str(e)}",
                extra={
                    "molecule_id": molecule_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise PolicyEvaluationError(
                f"Unexpected policy evaluation error: {type(e).__name__}: {str(e)}"
            ) from e

        state.set_decision(decision)

        total_execution_time_ms = self._compute_total_execution_time(state)
        self.logger.info(
            f"Triage run complete for molecule_id={molecule_id}",
            extra={
                "molecule_id": molecule_id,
                "decision_decision_type": decision.decision_type,
                "total_tools": len(self.tools),
                "executed_tools": len(state.tool_results),
                "failed_tools": self._count_failed_tools(state),
                "total_execution_time_ms": total_execution_time_ms,
                "early_termination": state.is_terminated()
            }
        )

        return state

    # ------------------------------------------------------------------
    # Private helpers — all preserved from Week 3, no changes
    # ------------------------------------------------------------------

    def _get_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _compute_elapsed_time_ms(self, start_timestamp: str) -> float:
        start_dt = datetime.fromisoformat(start_timestamp)
        end_dt = datetime.now(timezone.utc)
        return round((end_dt - start_dt).total_seconds() * 1000, 2)

    def _compute_total_execution_time(self, state: AgentState) -> float:
        start_time = state.execution_start_time
        end_time = state.get_decision_timestamp()
        if start_time is None or end_time is None:
            return 0.0
        if isinstance(start_time, str):
            start_dt = datetime.fromisoformat(start_time)
        else:
            start_dt = start_time
        if isinstance(end_time, str):
            end_dt = datetime.fromisoformat(end_time)
        else:
            end_dt = end_time
        return (end_dt - start_dt).total_seconds() * 1000

    def _validate_tool_result(self, result, tool_name: str) -> None:
        if result is None:
            raise StateValidationError(
                f"Tool {tool_name} returned None; expected ToolResult or dict"
            )

    def _validate_decision(self, decision: 'Decision') -> None:
        if decision is None:
            raise StateValidationError("PolicyEngine returned None decision")
        if not hasattr(decision, 'decision_type'):
            raise StateValidationError("Decision missing required 'decision_type' field")

    def _count_failed_tools(self, state: 'AgentState') -> int:
        failed = 0
        for result in state.tool_results.values():
            if isinstance(result, ToolResult) and result.status == ToolExecutionStatus.FAILURE:
                failed += 1
        return failed