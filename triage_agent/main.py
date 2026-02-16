"""
main.py

Entry point for the TRI_FLAG agentic research system.

This script serves as an architectural smoke test, demonstrating that the
core components (TriageAgent, PolicyEngine, Tools) can be wired together
and executed end-to-end. It contains no domain logic and uses placeholder
inputs suitable for Week 2-3 architectural validation.

Usage:
    python main.py

Expected behavior:
    - Logs system initialization
    - Instantiates agent, policy engine, and tool registry
    - Executes a minimal triage workflow
    - Logs the final decision
    - Exits cleanly

Author: TRI_FLAG Research Team
Week: 3 (Chemical Validity Checking)
"""

import logging
import sys
from typing import List

from agent.agent_state import AgentState
from agent.triage_agent import TriageAgent
from policies.policy_engine import PolicyEngine
from tools.base_tool import Tool
from tools.validity_tool import ValidityTool


def configure_logging() -> None:
    """
    Configure structured logging for traceability and debugging.
    
    Sets up INFO-level logging with timestamps, module names, and log levels
    to enable full execution tracing for reproducibility analysis.
    
    Rationale:
        - INFO level is appropriate for architectural validation
        - Timestamps enable reproducibility analysis
        - Module names help trace execution flow
        - Stream handler ensures visibility in development and CI/CD
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging configured for TRI_FLAG system")


def initialize_tools() -> List[Tool]:
    """
    Initialize the tool registry.
    
    Week 3: ValidityTool is registered FIRST to ensure
    invalid molecules are rejected before expensive computations.
    
    Returns:
        List of tools in execution order
    
    Rationale:
        - Explicit function makes future tool registration trivial
        - Returning empty list validates that agent handles zero tools
        - Type annotation ensures architectural contract is maintained
        
    Future usage (Week 4+):
        tools = [
            ValidityTool(),  # Always first
            SAScoreTool(),
            SimilarityTool(),
        ]
    """
    tools: List[Tool] = [
        ValidityTool(),  # MUST be first
        # Future tools will go here (SA Score, Similarity, etc.)
    ]
    logging.info(f"Initialized {len(tools)} tools")
    return tools


def initialize_policy_engine() -> PolicyEngine:
    """
    Initialize the policy evaluation engine.
    
    Returns:
        Configured PolicyEngine instance
    
    Rationale:
        - Encapsulates policy engine construction
        - Future configuration (e.g., risk thresholds) can be added here
        - Logging confirms successful initialization
        - Provides clear extension point for policy customization
    """
    policy_engine = PolicyEngine()
    logging.info("PolicyEngine initialized")
    return policy_engine


def initialize_agent(tools: List[Tool], policy_engine: PolicyEngine) -> TriageAgent:
    """
    Initialize the triage agent with dependencies.
    
    Args:
        tools: List of available evaluation tools
        policy_engine: Policy evaluation engine
    
    Returns:
        Configured TriageAgent instance
    
    Rationale:
        - Explicit dependency injection makes testing easier
        - Separates object construction from usage
        - Documents required dependencies clearly
        - Agent owns orchestration logic, not construction
        - Logger is created specifically for the agent's use
    """
    # Create a logger specifically for the TriageAgent
    agent_logger = logging.getLogger("agent.triage_agent")
    
    agent = TriageAgent(
        tools=tools,
        policy_engine=policy_engine,
        logger=agent_logger
    )
    logging.info("TriageAgent initialized with dependencies")
    return agent


def run_smoke_test(agent: TriageAgent) -> None:
    """
    Execute a minimal workflow to validate system wiring.
    
    Args:
        agent: Configured TriageAgent instance
    
    Raises:
        Any exception from agent.run() (fail-fast behavior for Week 3)
    
    Rationale:
        - Uses valid SMILES for Week 3 chemistry validation
        - Validates that agent.run() executes without exceptions
        - Logs outcome for traceability
        - Does not validate correctness of decision (that's a unit test concern)
        - Exceptions propagate for visibility during development
    """
    logging.info("=" * 60)
    logging.info("Starting architectural smoke test")
    logging.info("=" * 60)
    
    # Week 3: Use valid SMILES for chemistry validation
    molecule_id = "TEST_MOLECULE_001"
    raw_input = "CCO"  # Ethanol - simple valid molecule
    
    logging.info(f"Running triage for molecule_id='{molecule_id}'")
    
    # Execute the full pipeline
    # Note: We allow exceptions to propagate (fail-fast for Week 3)
    decision = agent.run(molecule_id=molecule_id, raw_input=raw_input)
    
    # Log the outcome using Decision.__str__() implementation
    logging.info(f"Triage completed. Decision: {decision}")
    logging.info("=" * 60)
    logging.info("Smoke test PASSED")
    logging.info("=" * 60)


def main() -> None:
    """
    Main entry point for TRI_FLAG system.
    
    Orchestrates:
        1. Logging configuration
        2. Component initialization (tools, policy engine, agent)
        3. Smoke test execution
        4. Clean exit
    
    This function remains minimal and focused on orchestration.
    All domain logic belongs in agent, tools, or policy modules.
    
    Error handling:
        - Exceptions propagate to caller (fail-fast for Week 3)
        - Full stack traces visible for debugging
        - No graceful degradation at this stage
    """
    # Step 1: Configure logging first for visibility
    configure_logging()
    
    # Step 2: Initialize components in dependency order
    logging.info("Initializing TRI_FLAG components...")
    tools = initialize_tools()
    policy_engine = initialize_policy_engine()
    agent = initialize_agent(tools=tools, policy_engine=policy_engine)
    
    # Step 3: Execute smoke test
    run_smoke_test(agent)
    
    # Step 4: Clean exit
    logging.info("TRI_FLAG system execution complete")


if __name__ == "__main__":
    """
    Module guard ensures main() only runs when executed directly.
    
    This allows main.py to be imported without side effects,
    which is essential for:
        - Testing frameworks (pytest can import without execution)
        - Interactive debugging (import in REPL)
        - Documentation generation tools
        - Future CLI wrappers
    """
    main()