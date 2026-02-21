"""
main.py

Entry point for the TRI_FLAG agentic research system.

This script serves as an architectural smoke test, demonstrating that the
core components (TriageAgent, PolicyEngine, Tools) can be wired together
and executed end-to-end.

Usage:
    python main.py

Expected behavior:
    - Logs system initialization
    - Instantiates agent, policy engine, and tool registry
    - Executes a minimal triage workflow (ethanol, CCO)
    - Logs the final decision
    - Exits cleanly

Performance note (Week 5):
    With SimilarityTool registered, `python main.py` takes ~3-8 seconds
    due to live ChEMBL and PubChem API round trips. This is expected — the
    trade-off for live database coverage. For offline testing, use the
    unit test suite which mocks all network calls.

Author: TRI_FLAG Research Team
Week: 5 (Similarity / IP-Risk Screening)
"""

import logging
import sys
from typing import List

from agent.agent_state import AgentState
from agent.triage_agent import TriageAgent
from policies.policy_engine import PolicyEngine
from tools.base_tool import Tool
from tools.validity_tool import ValidityTool
from tools.sa_score_tool import SAScoreTool
from tools.similarity_tool import SimilarityTool


def configure_logging() -> None:
    """
    Configure structured logging for traceability and debugging.

    Sets up INFO-level logging with timestamps, module names, and log levels
    to enable full execution tracing for reproducibility analysis.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.info("Logging configured for TRI_FLAG system")


def initialize_tools() -> List[Tool]:
    """
    Initialize the tool registry in execution order.

    Tool order (enforced):
        1. ValidityTool    — Week 3: invalid molecules exit immediately
        2. SAScoreTool     — Week 4: SA > 7 exits immediately; SA 6-7 → FLAG
        3. SimilarityTool  — Week 5: Tanimoto >= 0.85 → FLAG (never exits early)

    Performance note:
        SimilarityTool adds ~3-8s to main.py due to live API calls.
        Unit tests use mocked network calls and run in <1s.

    Returns:
        Ordered list of tools
    """
    tools: List[Tool] = [
        ValidityTool(),      # MUST be first — rejects invalid chemistry
        SAScoreTool(),       # MUST be second — rejects un-synthesizable molecules
        SimilarityTool(),    # Week 5: IP-risk screening via ChEMBL + PubChem
        # Future tools will go here (ADMET, selectivity, etc.)
    ]
    logging.info("Initialized %d tools: %s", len(tools), [t.name for t in tools])
    return tools


def initialize_policy_engine() -> PolicyEngine:
    """
    Initialize the policy evaluation engine.

    Returns:
        Configured PolicyEngine instance
    """
    policy_engine = PolicyEngine()
    logging.info("PolicyEngine initialized")
    return policy_engine


def initialize_agent(
    tools: List[Tool], policy_engine: PolicyEngine
) -> TriageAgent:
    """
    Initialize the triage agent with dependencies.

    Args:
        tools: Ordered list of available evaluation tools
        policy_engine: Policy evaluation engine

    Returns:
        Configured TriageAgent instance
    """
    agent_logger = logging.getLogger("agent.triage_agent")
    agent = TriageAgent(
        tools=tools,
        policy_engine=policy_engine,
        logger=agent_logger,
    )
    logging.info("TriageAgent initialized with %d tools", len(tools))
    return agent


def run_smoke_test(agent: TriageAgent) -> None:
    """
    Execute a minimal workflow to validate system wiring.

    Week 5: Ethanol (CCO) is used because it is:
        - Chemically valid (passes ValidityTool)
        - Trivially synthesizable (SA ≈ 1.0, passes SAScoreTool)
        - Well-known in ChEMBL/PubChem (likely similar results from APIs)

    Note on expected output:
        Ethanol is a widely registered compound; SimilarityTool may FLAG it
        due to high Tanimoto similarity against simple alcohols in ChEMBL.
        This is expected behavior — the FLAG confirms API connectivity and
        correct threshold application.

    Args:
        agent: Configured TriageAgent instance

    Raises:
        Any exception from agent.run() (fail-fast behavior)
    """
    logging.info("=" * 60)
    logging.info("Starting architectural smoke test (Week 5)")
    logging.info("=" * 60)

    molecule_id = "TEST_MOLECULE_001"
    raw_input = "CCO"  # Ethanol — simple valid molecule with known DB entries

    logging.info(
        "Running triage for molecule_id='%s' (SMILES: %s)",
        molecule_id,
        raw_input,
    )
    logging.info(
        "Note: SimilarityTool will make live API calls — expect ~3-8s latency."
    )

    # Execute the full pipeline (exceptions propagate — fail-fast)
    state = agent.run(molecule_id=molecule_id, raw_input=raw_input)
    decision = state.decision

    logging.info("Triage completed. Decision: %s", decision)
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

    Error handling:
        Exceptions propagate to caller (fail-fast for development).
    """
    configure_logging()

    logging.info("Initializing TRI_FLAG components (Week 5)...")
    tools = initialize_tools()
    policy_engine = initialize_policy_engine()
    agent = initialize_agent(tools=tools, policy_engine=policy_engine)

    run_smoke_test(agent)

    logging.info("TRI_FLAG system execution complete")


if __name__ == "__main__":
    main()