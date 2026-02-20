"""
main.py

Entry point for the TRI_FLAG agentic research system.

Week 3: ValidityTool registered — invalid molecules discarded early
Week 4: SAScoreTool registered — synthetically intractable molecules
        discarded, challenging molecules flagged and continued
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


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.info("Logging configured for TRI_FLAG system")


def initialize_tools() -> List[Tool]:
    """
    Initialize the tool registry in execution order.

    Order is critical:
        1. ValidityTool  — Must be first. Invalid chemistry terminates here.
        2. SAScoreTool   — Runs only on valid molecules.
                          SA > 7  -> DISCARD (terminate)
                          SA 6-7  -> FLAG (annotate, continue)
                          SA < 6  -> PASS (continue)
        # Week 5:
        # 3. SimilarityTool — novelty / IP risk check
    """
    tools: List[Tool] = [
        ValidityTool(),
        SAScoreTool(),
    ]
    logging.info(f"Initialized {len(tools)} tools: {[t.name for t in tools]}")
    return tools


def initialize_policy_engine() -> PolicyEngine:
    policy_engine = PolicyEngine()
    logging.info("PolicyEngine initialized")
    return policy_engine


def initialize_agent(tools: List[Tool], policy_engine: PolicyEngine) -> TriageAgent:
    agent_logger = logging.getLogger("agent.triage_agent")
    agent = TriageAgent(
        tools=tools,
        policy_engine=policy_engine,
        logger=agent_logger
    )
    logging.info("TriageAgent initialized with dependencies")
    return agent


def run_smoke_test(agent: TriageAgent) -> None:
    logging.info("=" * 60)
    logging.info("Starting Week 4 architectural smoke test")
    logging.info("=" * 60)

    test_cases = [
        ("TEST_EASY_001",    "CCO",                             "Ethanol — expect PASS"),
        ("TEST_MEDIUM_001",  "CC(=O)Oc1ccccc1C(=O)O",          "Aspirin — expect PASS"),
        ("TEST_INVALID_001", "C(C)(C)(C)(C)C",                  "Invalid valence — expect DISCARD"),
        ("TEST_COMPLEX_001",
         "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
         "Cholesterol — may FLAG or DISCARD"),
    ]

    for molecule_id, smiles, description in test_cases:
        logging.info(f"\n  --- {description} ---")
        state = agent.run(molecule_id=molecule_id, raw_input=smiles)

        decision = state.decision
        sa_result = state.tool_results.get("SAScoreTool")

        logging.info(f"  Decision: {decision}")
        if sa_result and isinstance(sa_result, dict) and sa_result.get("sa_score"):
            logging.info(
                f"  SA Score: {sa_result['sa_score']:.2f} "
                f"({sa_result.get('synthesizability_category', '?')})"
            )
        if state.is_flagged():
            logging.info(f"  Flags: {state.get_flags()}")
        if state.is_terminated():
            logging.info(f"  Terminated: {state.get_termination_reason()}")

    logging.info("=" * 60)
    logging.info("Smoke test PASSED")
    logging.info("=" * 60)


def main() -> None:
    configure_logging()
    logging.info("Initializing TRI_FLAG components...")
    tools = initialize_tools()
    policy_engine = initialize_policy_engine()
    agent = initialize_agent(tools=tools, policy_engine=policy_engine)
    run_smoke_test(agent)
    logging.info("TRI_FLAG system execution complete")


if __name__ == "__main__":
    main()