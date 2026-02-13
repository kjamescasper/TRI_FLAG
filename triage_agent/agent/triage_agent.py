"""
TriageAgent: Central orchestration controller for molecular triage workflow.

Week 2 Implementation - Core Architecture
This module implements the deterministic, state-centric agent that coordinates
tool execution, maintains provenance, and delegates decision-making.

Design Principles:
- Tools execute sequentially in fixed order (no parallelism in Week 2)
- Tools are stateless; AgentState is the single source of truth
- Tool failures are non-terminal by default; execution continues with recorded errors
- State becomes immutable after decision is set
- Early termination allowed via explicit state signals only
- PolicyEngine evaluates completed state; does not control execution flow
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
    
    Tool Execution Semantics (Week 2):
    - Tools are executed sequentially in the order provided at construction
    - No dependency management or parallel execution
    - Tool failures are non-terminal by default (execution continues)
    - Early termination is supported via explicit state signals
    
    State Management:
    - AgentState is initialized at run start and owned by this agent
    - State transitions to immutable after decision is set
    - All tool outputs and execution metadata are recorded in state
    
    Attributes:
        tools: Ordered sequence of Tool instances to execute
        policy_engine: PolicyEngine instance for decision evaluation
        logger: Logging interface for execution tracking
    """
    
    def __init__(
        self,
        tools: List['Tool'],
        policy_engine: 'PolicyEngine',
        logger: 'Logger'
    ):
        """
        Initialize the triage agent with required dependencies.
        
        Args:
            tools: Ordered list of Tool instances. Execution order is list order.
                  Tools are executed sequentially in the order provided.
            policy_engine: PolicyEngine instance for state evaluation
            logger: Logger instance for execution and provenance tracking
        
        Raises:
            ValueError: If tools list is empty or contains duplicate tool names
        """
        if not tools:
            logger.warning("TriageAgent initialized with 0 tools - architectural validation mode")
        
        # Validate no duplicate tool names
        tool_names = [tool.get_name() for tool in tools]
        if len(tool_names) != len(set(tool_names)):
            duplicates = [name for name in tool_names if tool_names.count(name) > 1]
            raise ValueError(f"Duplicate tool names found: {set(duplicates)}")
        
        self.tools = tools
        self.policy_engine = policy_engine
        self.logger = logger
    
    def run(self, molecule_id: str, raw_input: dict) -> 'AgentState':
        """
        Execute the complete triage workflow for a single molecule.
        
        Orchestrates tool execution, state updates, policy evaluation, and
        decision finalization. Maintains full execution provenance for
        reproducibility and debugging.
        
        Execution Flow:
        1. Initialize AgentState with molecule_id and raw_input
        2. Execute tools sequentially in fixed order
        3. Record all tool results (success or failure) in state
        4. Check for early termination signals between tools
        5. Delegate decision-making to PolicyEngine
        6. Finalize and return immutable state
        
        Args:
            molecule_id: Unique identifier for the molecule being triaged
            raw_input: Unprocessed input data containing molecule information.
                      Structure and validation handled by upstream components.
        
        Returns:
            AgentState: Complete final state containing all tool results,
                       execution metadata, and final decision. State is
                       immutable after this method returns.
        
        Raises:
            StateValidationError: If state becomes invalid during execution
            PolicyEvaluationError: If policy evaluation fails critically
            ToolExecutionError: If a tool with TERMINAL failure mode fails
        """
        # Initialize state as single source of truth
        state = AgentState(
            molecule_id=molecule_id,
            raw_input=raw_input,
            execution_start_time=datetime.now(timezone.utc)  # Adds time tracking
        )
        
        self.logger.info(
            f"Starting triage run for molecule_id={molecule_id}",
            extra={"molecule_id": molecule_id}
        )
        
        # Execute tools in deterministic sequential order
        for tool in self.tools:
            # Check for early termination signal
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
            
            tool_name = tool.get_name()
            tool_start_time = self._get_timestamp()
            
            self.logger.debug(
                f"Executing tool: {tool_name}",
                extra={"molecule_id": molecule_id, "tool": tool_name}
            )
            
            try:
                # Tool executes and returns result (tool is stateless)
                result = tool.run(state)
                
                # Calculate execution time
                execution_time_ms = self._compute_elapsed_time_ms(tool_start_time)
                
                # Validate result structure
                self._validate_tool_result(result, tool_name)
                
                # Ensure result has execution time
                if isinstance(result, ToolResult) and result.execution_time_ms is None:
                    result.execution_time_ms = execution_time_ms
                
                # Update state with tool output
                state.add_tool_result(
                    tool_name=tool_name,
                    result=result
                )
                # ═══════════════════════════════════════════════════════════════
            # WEEK 3 ADDITION: Check for validity tool failure
            # ═══════════════════════════════════════════════════════════════
                if tool_name == "ValidityTool":
                # Extract validity result from tool output
                    if isinstance(result, ToolResult):
                        validity_data = result.data
                    else:
                        validity_data = result
                
                    is_valid = validity_data.get('is_valid', False)
                
                    if not is_valid:
                        # Molecule is chemically invalid - terminate early
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
                        # Break out of tool loop - don't run any more tools
                        break
                    else:
                        # Molecule is valid - log and continue
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
            # ═══════════════════════════════════════════════════════════════
            # END WEEK 3 ADDITION
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
                # Handle explicit tool failures
                execution_time_ms = self._compute_elapsed_time_ms(tool_start_time)
                
                error_result = ToolResult(
                    tool_name=tool_name,
                    status=ToolExecutionStatus.FAILURE,
                    error_message=str(e),
                    execution_time_ms=execution_time_ms
                )
                
                state.add_tool_result(
                    tool_name=tool_name,
                    result=error_result
                )
                
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
                
                # Check tool failure mode policy
                failure_mode = tool.get_failure_mode()
                if failure_mode == ToolFailureMode.TERMINAL:
                    self.logger.critical(
                        f"Tool {tool_name} has TERMINAL failure mode; aborting execution",
                        extra={
                            "molecule_id": molecule_id,
                            "tool": tool_name,
                            "failure_mode": failure_mode.value
                        }
                    )
                    raise
                
                # NON_TERMINAL: continue execution with recorded failure
                
            except Exception as e:
                # Handle unexpected failures
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
                
                # Record unexpected error in state before re-raising
                error_result = ToolResult(
                    tool_name=tool_name,
                    status=ToolExecutionStatus.FAILURE,
                    error_message=f"Unexpected error: {type(e).__name__}: {str(e)}",
                    execution_time_ms=execution_time_ms
                )
                
                state.add_tool_result(
                    tool_name=tool_name,
                    result=error_result
                )
                
                raise
        
        # Mark tool execution phase complete
        #, no tools to implement, put back in later: state.set_tools_complete(timestamp=self._get_timestamp())
        
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
        
        # Delegate decision-making to policy engine
        try:
            self.logger.debug(
                f"Invoking PolicyEngine for molecule_id={molecule_id}",
                extra={"molecule_id": molecule_id}
            )
            
            decision = self.policy_engine.evaluate(state)
            
            # Validate decision object
            self._validate_decision(decision)
            
            self.logger.info(
                f"Policy evaluation complete: {decision.decision_type}",
                extra={
                    "molecule_id": molecule_id,
                    "decision_decision_type": decision.decision_type,
                    "decision_confidence": getattr(decision, 'confidence', None)
                }
            )
            
        except PolicyEvaluationError as e:
            self.logger.error(
                f"Policy evaluation failed: {str(e)}",
                extra={
                    "molecule_id": molecule_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
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
        
        # Store final decision in state (state becomes immutable after this)
        state.set_decision(decision)
        
        # Compute and log final execution statistics
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
    
    def _get_timestamp(self) -> str:
        """
        Generate ISO-8601 timestamp for provenance tracking.
        
        Uses UTC timezone for consistency across distributed systems.
        
        Returns:
            ISO-8601 formatted timestamp string with timezone
        """
        return datetime.now(timezone.utc).isoformat()
    
    def _compute_elapsed_time_ms(self, start_timestamp: str) -> float:
        """
        Calculate elapsed time from start timestamp to now.
        
        Args:
            start_timestamp: ISO-8601 formatted start time
        
        Returns:
            Elapsed time in milliseconds
        """
        start_dt = datetime.fromisoformat(start_timestamp)
        end_dt = datetime.now(timezone.utc)
        elapsed = (end_dt - start_dt).total_seconds() * 1000
        return round(elapsed, 2)
    
    def _compute_total_execution_time(self, state: AgentState) -> float:
    ### Compute total execution time in milliseconds.
    
    # Args:
    #    state: AgentState with execution timestamps
    
    # Returns:
    #    Execution time in milliseconds

        start_time = state.execution_start_time
        end_time = state.get_decision_timestamp()
    
        if start_time is None or end_time is None:
            return 0.0
    
    # Handle both datetime objects and ISO strings
        if isinstance(start_time, str):
            start_dt = datetime.fromisoformat(start_time)
        else:
            start_dt = start_time
    
        if isinstance(end_time, str):
            end_dt = datetime.fromisoformat(end_time)
        else:
            end_dt = end_time
    
    # Compute difference in milliseconds
        time_diff = (end_dt - start_dt).total_seconds() * 1000
        return time_diff
    
    def _validate_tool_result(self, result: any, tool_name: str) -> None:
        """
        Validate tool result structure and content.
        
        Args:
            result: Tool execution result to validate
            tool_name: Name of tool that produced result
        
        Raises:
            StateValidationError: If result is invalid
        """
        # TODO: Implement comprehensive result validation
        # For Week 2: basic type check
        if result is None:
            raise StateValidationError(
                f"Tool {tool_name} returned None; expected ToolResult or dict"
            )
    
    def _validate_decision(self, decision: 'Decision') -> None:
        """
        Validate decision object structure and content.
        
        Args:
            decision: Decision object to validate
        
        Raises:
            StateValidationError: If decision is invalid
        """
        # TODO: Implement comprehensive decision validation
        # For Week 2: basic existence check
        if decision is None:
            raise StateValidationError("PolicyEngine returned None decision")
        
        if not hasattr(decision, 'decision_type'):
            raise StateValidationError("Decision missing required 'decision_type' field")
    
    def _count_failed_tools(self, state: 'AgentState') -> int:
        """
        Count number of failed tools in execution.
        
        Args:
            state: Agent state with tool results
        
        Returns:
            Number of tools with FAILURE status
        """
        failed_count = 0
        for result in state.tool_results.values():
            if isinstance(result, ToolResult) and result.status == ToolExecutionStatus.FAILURE:
                failed_count += 1
        return failed_count