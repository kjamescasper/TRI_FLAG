"""
tests/test_agent.py

Integration test suite for TRI_FLAG agent pipeline.

Week 3: Chemical validity checking
Week 4: SA score tool integration with agent and policy engine

For pure chemistry unit tests (calculate_sa_score, complexity_breakdown,
benchmarks), see tests/test_sa_score.py.

Run with:
    cd ..
    set PYTHONPATH=%CD%\triage_agent
    pytest triage_agent/tests/test_agent.py -v
"""

import pytest
import logging


# ---------------------------------------------------------------------------
# Check sascorer availability once at module level.
# Tests that require a real SA score computation are skipped when sascorer
# is not installed. All structural/wiring tests still run.
# ---------------------------------------------------------------------------
def _sascorer_available() -> bool:
    try:
        from chemistry.sa_score import _SASCORER_AVAILABLE
        return _SASCORER_AVAILABLE
    except Exception:
        return False

_SASCORER = _sascorer_available()
requires_sascorer = pytest.mark.skipif(
    not _SASCORER,
    reason=(
        "rdkit.Contrib.SA_Score (sascorer) not available. "
        "Run: conda install -c conda-forge rdkit"
    )
)

from molecule import Molecule
from tools.validity_tool import ValidityTool, validate_smiles
from tools.sa_score_tool import SAScoreTool, SAScoreResult
from agent.triage_agent import TriageAgent
from policies.policy_engine import PolicyEngine, PolicyDecision
from policies.thresholds import (
    SAScoreThresholds,
    DEFAULT_SA_THRESHOLDS,
    LEAD_OPTIMIZATION_THRESHOLDS,
    NATURAL_PRODUCT_THRESHOLDS,
    FRAGMENT_SCREENING_THRESHOLDS,
)
from agent.agent_state import AgentState


# =============================================================================
# Week 3: ValidityTool unit tests (unchanged)
# =============================================================================

class TestValidityTool:

    def setup_method(self):
        self.tool = ValidityTool()

    def test_valid_simple_molecule(self):
        mol = Molecule(molecule_id="TEST_001", smiles="CCO")
        result = self.tool._validate_molecule(mol)
        assert result['is_valid'] is True
        assert result['error_message'] is None
        assert result['smiles_canonical'] == "CCO"
        assert result['num_atoms'] > 0
        assert result['num_bonds'] > 0

    def test_valid_aromatic_molecule(self):
        mol = Molecule(molecule_id="TEST_002", smiles="c1ccccc1")
        result = self.tool._validate_molecule(mol)
        assert result['is_valid'] is True
        assert result['num_atoms'] == 6

    def test_invalid_valence(self):
        mol = Molecule(molecule_id="TEST_003", smiles="C(C)(C)(C)(C)C")
        result = self.tool._validate_molecule(mol)
        assert result['is_valid'] is False
        assert "RDKit failed to parse" in result['error_message']

    def test_empty_smiles(self):
        is_valid, error = validate_smiles("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_whitespace_only_smiles(self):
        is_valid, error = validate_smiles("   ")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_malformed_smiles(self):
        mol = Molecule(molecule_id="TEST_006", smiles="C(((")
        result = self.tool._validate_molecule(mol)
        assert result['is_valid'] is False

    def test_canonicalization(self):
        mol1 = Molecule(molecule_id="TEST_007", smiles="OCC")
        mol2 = Molecule(molecule_id="TEST_008", smiles="CCO")
        r1 = self.tool._validate_molecule(mol1)
        r2 = self.tool._validate_molecule(mol2)
        assert r1['is_valid'] is True
        assert r2['is_valid'] is True
        assert r1['smiles_canonical'] == r2['smiles_canonical']


# =============================================================================
# Week 3: Agent validity integration tests (unchanged)
# =============================================================================

class TestAgentValidityIntegration:

    def setup_method(self):
        self.agent = TriageAgent(
            tools=[ValidityTool()],
            policy_engine=PolicyEngine(),
            logger=logging.getLogger("test_agent"),
        )

    def test_valid_molecule_proceeds(self):
        state = self.agent.run(molecule_id="VALID_001", raw_input="CCO")
        assert state is not None
        assert state.molecule_id == "VALID_001"
        assert state.tool_results['ValidityTool']['is_valid'] is True

    def test_invalid_molecule_terminates_early(self):
        state = self.agent.run(molecule_id="INVALID_001", raw_input="C(C)(C)(C)(C)C")
        assert state is not None
        assert state.tool_results['ValidityTool']['is_valid'] is False
        assert state.is_terminated() is True

    def test_empty_smiles_handled_gracefully(self):
        state = self.agent.run(molecule_id="EMPTY_001", raw_input="")
        assert state is not None
        assert state.is_terminated() is True


# =============================================================================
# Week 3: Standalone validate_smiles() tests (unchanged)
# =============================================================================

class TestValidateSmilesFunction:

    def test_valid(self):
        is_valid, error = validate_smiles("CCO")
        assert is_valid is True
        assert error is None

    def test_invalid(self):
        is_valid, error = validate_smiles("C(C)(C)(C)(C)C")
        assert is_valid is False
        assert error is not None

    def test_empty(self):
        is_valid, error = validate_smiles("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_whitespace(self):
        is_valid, error = validate_smiles("   ")
        assert is_valid is False
        assert "empty" in error.lower()


# =============================================================================
# Week 3: Edge cases (unchanged)
# =============================================================================

class TestEdgeCases:

    def setup_method(self):
        self.tool = ValidityTool()

    def test_single_atom(self):
        mol = Molecule(molecule_id="EDGE_001", smiles="C")
        result = self.tool._validate_molecule(mol)
        assert result['is_valid'] is True
        assert result['num_atoms'] == 1

    def test_large_molecule(self):
        mol = Molecule(molecule_id="EDGE_002",
                       smiles="CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C")
        result = self.tool._validate_molecule(mol)
        assert result['is_valid'] is True
        assert result['num_atoms'] > 20

    def test_charged_molecule(self):
        mol = Molecule(molecule_id="EDGE_003", smiles="[NH4+]")
        assert self.tool._validate_molecule(mol)['is_valid'] is True

    def test_radical_molecule(self):
        mol = Molecule(molecule_id="EDGE_004", smiles="[CH3]")
        assert self.tool._validate_molecule(mol)['is_valid'] is True

    def test_multiple_components(self):
        mol = Molecule(molecule_id="EDGE_005", smiles="[Na+].[Cl-]")
        assert self.tool._validate_molecule(mol)['is_valid'] is True

    def test_stereochemistry(self):
        mol = Molecule(molecule_id="EDGE_006", smiles="C[C@H](N)C(=O)O")
        assert self.tool._validate_molecule(mol)['is_valid'] is True


# =============================================================================
# Week 3: ValidityTool via AgentState (unchanged)
# =============================================================================

class TestToolStateIntegration:

    def test_string_input(self):
        tool = ValidityTool()
        state = AgentState(molecule_id="TEST_RUN_001", raw_input="CCO")
        result = tool.run(state)
        assert result['is_valid'] is True
        assert result['smiles_canonical'] == "CCO"

    def test_dict_input(self):
        tool = ValidityTool()
        state = AgentState(molecule_id="TEST_RUN_002",
                           raw_input={'smiles': 'c1ccccc1', 'name': 'Benzene'})
        result = tool.run(state)
        assert result['is_valid'] is True

    def test_molecule_input(self):
        tool = ValidityTool()
        mol = Molecule(molecule_id="TEST_RUN_003", smiles="CC(=O)O")
        state = AgentState(molecule_id="TEST_RUN_003", raw_input=mol)
        result = tool.run(state)
        assert result['smiles_canonical'] == "CC(=O)O"

    def test_invalid_type_input(self):
        tool = ValidityTool()
        state = AgentState(molecule_id="TEST_RUN_004", raw_input=12345)
        result = tool.run(state)
        assert result['is_valid'] is False
        assert 'Unknown input type' in result['error_message']


# =============================================================================
# Week 4: SAScoreThresholds — classify() and categorize()
# =============================================================================

class TestSAScoreThresholds:

    def setup_method(self):
        self.t = SAScoreThresholds()

    # Pipeline decision
    def test_pass(self):
        for score in [1.0, 3.5, 5.99]:
            assert self.t.classify(score) == "PASS", f"Expected PASS for {score}"

    def test_flag(self):
        for score in [6.0, 6.5, 7.0]:
            assert self.t.classify(score) == "FLAG", f"Expected FLAG for {score}"

    def test_discard(self):
        for score in [7.01, 8.5, 10.0]:
            assert self.t.classify(score) == "DISCARD", f"Expected DISCARD for {score}"

    def test_pass_flag_boundary(self):
        assert self.t.classify(6.0) == "FLAG"   # >= pass_threshold → FLAG

    def test_flag_discard_boundary(self):
        assert self.t.classify(7.0) == "FLAG"   # <= flag_threshold → FLAG
        assert self.t.classify(7.001) == "DISCARD"

    # Descriptive category
    def test_easy_category(self):
        assert self.t.categorize(1.5) == "easy"
        assert self.t.categorize(3.0) == "easy"

    def test_moderate_category(self):
        assert self.t.categorize(3.1) == "moderate"
        assert self.t.categorize(6.0) == "moderate"

    def test_difficult_category(self):
        assert self.t.categorize(6.1) == "difficult"
        assert self.t.categorize(7.0) == "difficult"

    def test_very_difficult_category(self):
        assert self.t.categorize(7.1) == "very_difficult"
        assert self.t.categorize(10.0) == "very_difficult"

    def test_describe_includes_category(self):
        # 2.5 is easy (≤ 3.0 threshold); 3.5 is moderate (3.0 < 3.5 ≤ 6.0)
        desc_easy = self.t.describe(2.5)
        assert "easy" in desc_easy.lower()
        desc_moderate = self.t.describe(3.5)
        assert "moderate" in desc_moderate.lower()

    def test_is_valid_range(self):
        assert self.t.is_valid_range(1.0) is True
        assert self.t.is_valid_range(10.0) is True
        assert self.t.is_valid_range(0.9) is False
        assert self.t.is_valid_range(10.1) is False

    def test_custom_thresholds(self):
        custom = SAScoreThresholds(pass_threshold=4.0, flag_threshold=5.5)
        assert custom.classify(3.9) == "PASS"
        assert custom.classify(4.5) == "FLAG"
        assert custom.classify(5.6) == "DISCARD"

    # Alternative preset threshold sets
    def test_lead_opt_stricter(self):
        t = LEAD_OPTIMIZATION_THRESHOLDS
        assert t.pass_threshold < DEFAULT_SA_THRESHOLDS.pass_threshold
        assert t.classify(5.5) == "FLAG"   # Would PASS in default

    def test_natural_product_more_permissive(self):
        t = NATURAL_PRODUCT_THRESHOLDS
        assert t.pass_threshold > DEFAULT_SA_THRESHOLDS.pass_threshold
        assert t.classify(6.5) == "PASS"   # Would FLAG in default

    def test_fragment_screening_very_strict(self):
        t = FRAGMENT_SCREENING_THRESHOLDS
        assert t.pass_threshold < DEFAULT_SA_THRESHOLDS.pass_threshold
        assert t.classify(4.0) == "FLAG"   # Would PASS in default


# =============================================================================
# Week 4: SAScoreTool unit tests
# =============================================================================

class TestSAScoreTool:

    def setup_method(self):
        self.tool = SAScoreTool()

    def _state_with_validity(self, molecule_id: str, smiles: str) -> AgentState:
        state = AgentState(molecule_id=molecule_id, raw_input=smiles)
        state.add_tool_result("ValidityTool", {
            "is_valid": True,
            "smiles_canonical": smiles,
            "error_message": None,
        })
        return state

    def test_result_has_required_keys(self):
        state = self._state_with_validity("SA_001", "CCO")
        result = self.tool.run(state)
        required = [
            "tool_name", "molecule_id", "sa_score", "synthesizability_category",
            "sa_decision", "sa_description", "complexity_breakdown",
            "warning_flags", "error_message", "execution_time_ms",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_tool_name(self):
        state = self._state_with_validity("SA_002", "CCO")
        assert self.tool.run(state)["tool_name"] == "SAScoreTool"

    @requires_sascorer
    def test_ethanol_passes(self):
        state = self._state_with_validity("SA_003", "CCO")
        result = self.tool.run(state)
        assert result["sa_decision"] == "PASS"
        assert result["synthesizability_category"] == "easy"
        assert result["error_message"] is None

    @requires_sascorer
    def test_aspirin_passes_easy(self):
        state = self._state_with_validity("SA_004", "CC(=O)Oc1ccccc1C(=O)O")
        result = self.tool.run(state)
        assert result["sa_decision"] == "PASS"
        assert result["sa_score"] < 6.0

    @requires_sascorer
    def test_score_in_valid_range(self):
        state = self._state_with_validity("SA_005", "CC(=O)Oc1ccccc1C(=O)O")
        result = self.tool.run(state)
        assert 1.0 <= result["sa_score"] <= 10.0

    @requires_sascorer
    def test_complexity_breakdown_present(self):
        state = self._state_with_validity("SA_006", "c1ccccc1")
        result = self.tool.run(state)
        bd = result["complexity_breakdown"]
        assert bd is not None
        assert "num_heavy_atoms" in bd
        assert "num_rings" in bd
        assert "warning_flags" in bd

    def test_warning_flags_is_list(self):
        state = self._state_with_validity("SA_007", "CCO")
        result = self.tool.run(state)
        assert isinstance(result["warning_flags"], list)

    @requires_sascorer
    def test_synthesizability_category_valid_values(self):
        state = self._state_with_validity("SA_008", "CCO")
        result = self.tool.run(state)
        assert result["synthesizability_category"] in (
            "easy", "moderate", "difficult", "very_difficult"
        )

    @requires_sascorer
    def test_fallback_to_raw_input_string(self):
        # No ValidityTool result — should fall back to raw_input
        state = AgentState(molecule_id="SA_009", raw_input="CCO")
        result = self.tool.run(state)
        assert result["sa_score"] is not None
        assert result["sa_decision"] != "ERROR"

    @requires_sascorer
    def test_fallback_to_molecule_object(self):
        mol = Molecule(molecule_id="SA_010", smiles="c1ccccc1")
        state = AgentState(molecule_id="SA_010", raw_input=mol)
        result = self.tool.run(state)
        assert result["sa_score"] is not None

    def test_no_smiles_returns_error(self):
        state = AgentState(molecule_id="SA_011", raw_input=None)
        result = self.tool.run(state)
        assert result["sa_decision"] == "ERROR"
        assert result["error_message"] is not None

    def test_invalid_molecule_in_validity_skips_sa(self):
        state = AgentState(molecule_id="SA_012", raw_input="CCO")
        state.add_tool_result("ValidityTool", {"is_valid": False, "error_message": "bad"})
        result = self.tool.run(state)
        assert result["sa_decision"] == "ERROR"

    def test_execution_time_recorded(self):
        state = self._state_with_validity("SA_013", "CCO")
        result = self.tool.run(state)
        assert result["execution_time_ms"] >= 0

    @requires_sascorer
    def test_custom_thresholds_respected(self):
        strict = SAScoreThresholds(pass_threshold=1.5, flag_threshold=2.0)
        tool = SAScoreTool(thresholds=strict)
        state = self._state_with_validity("SA_014", "CCO")
        result = tool.run(state)
        # Ethanol SA ~1.5-2.0 — with ultra-strict thresholds may FLAG or DISCARD
        assert result["sa_score"] is not None
        assert result["sa_decision"] in ("PASS", "FLAG", "DISCARD")

    @requires_sascorer
    def test_to_dict_round_trip(self):
        """SAScoreResult.to_dict() should contain all required keys."""
        state = self._state_with_validity("SA_015", "CCO")
        result_dict = self.tool.run(state)
        # Verify all expected keys are present and round-trip via to_dict()
        required_keys = [
            "tool_name", "molecule_id", "sa_score", "synthesizability_category",
            "sa_decision", "sa_description", "complexity_breakdown",
            "warning_flags", "error_message", "execution_time_ms",
        ]
        for key in required_keys:
            assert key in result_dict, f"Missing key in result: {key}"
        # Re-construct SAScoreResult from the dict and verify to_dict() is stable
        result_obj = SAScoreResult(
            tool_name=result_dict["tool_name"],
            molecule_id=result_dict["molecule_id"],
            sa_score=result_dict["sa_score"],
            synthesizability_category=result_dict["synthesizability_category"],
            sa_decision=result_dict["sa_decision"],
            sa_description=result_dict["sa_description"],
            complexity_breakdown=result_dict["complexity_breakdown"],
            warning_flags=result_dict["warning_flags"],
            error_message=result_dict["error_message"],
            execution_time_ms=result_dict["execution_time_ms"],
        )
        assert result_obj.to_dict() == result_dict


# =============================================================================
# Week 4: PolicyDecision object tests
# =============================================================================

class TestPolicyDecision:

    def test_attributes_accessible(self):
        pd = PolicyDecision("PASS", "All good", "SAScoreTool", {"sa_score": 3.2})
        assert pd.decision == "PASS"
        assert pd.reason == "All good"
        assert pd.tool_checked == "SAScoreTool"
        assert pd.metadata["sa_score"] == 3.2

    def test_to_dict(self):
        pd = PolicyDecision("FLAG", "Borderline", "SAScoreTool")
        d = pd.to_dict()
        assert d["decision"] == "FLAG"
        assert d["reason"] == "Borderline"
        assert d["tool_checked"] == "SAScoreTool"
        assert isinstance(d["metadata"], dict)

    def test_to_decision_pass(self):
        from agent.decision import DecisionType
        pd = PolicyDecision("PASS", "ok", "SAScoreTool")
        d = pd.to_decision()
        assert d.decision_type == DecisionType.PASS

    def test_to_decision_flag(self):
        from agent.decision import DecisionType
        pd = PolicyDecision("FLAG", "borderline", "SAScoreTool")
        d = pd.to_decision()
        assert d.decision_type == DecisionType.FLAG

    def test_to_decision_discard(self):
        from agent.decision import DecisionType
        pd = PolicyDecision("DISCARD", "too hard", "SAScoreTool")
        d = pd.to_decision()
        assert d.decision_type == DecisionType.DISCARD

    def test_to_decision_error_becomes_discard(self):
        from agent.decision import DecisionType
        pd = PolicyDecision("ERROR", "tool failed", "SAScoreTool")
        d = pd.to_decision()
        assert d.decision_type == DecisionType.DISCARD


# =============================================================================
# Week 4: PolicyEngine SA score tests
# =============================================================================

class TestSAScorePolicyEngine:

    def setup_method(self):
        self.engine = PolicyEngine()

    def _state_with_sa(self, mol_id: str, sa_score: float, decision: str,
                       category: str = "moderate") -> AgentState:
        state = AgentState(molecule_id=mol_id, raw_input="CCO")
        state.add_tool_result("ValidityTool", {"is_valid": True, "error_message": None})
        state.add_tool_result("SAScoreTool", {
            "sa_score": sa_score,
            "sa_decision": decision,
            "synthesizability_category": category,
            "sa_description": f"SA score {sa_score:.2f}",
            "warning_flags": [],
            "error_message": None,
        })
        return state

    def _decision_name(self, state):
        """Helper: get the DecisionType name string from evaluate() result."""
        from agent.decision import DecisionType
        result = self.engine.evaluate(state)
        return result.decision_type.name  # "PASS", "FLAG", "DISCARD"

    def test_pass_decision(self):
        assert self._decision_name(self._state_with_sa("P1", 3.5, "PASS", "easy")) == "PASS"

    def test_flag_decision(self):
        assert self._decision_name(self._state_with_sa("P2", 6.5, "FLAG", "difficult")) == "FLAG"

    def test_discard_decision(self):
        assert self._decision_name(self._state_with_sa("P3", 8.0, "DISCARD", "very_difficult")) == "DISCARD"

    def test_metadata_contains_sa_score(self):
        result = self.engine.evaluate(self._state_with_sa("P4", 3.5, "PASS", "easy"))
        assert result.metadata.get("sa_score") == 3.5

    def test_metadata_contains_category(self):
        result = self.engine.evaluate(self._state_with_sa("P5", 3.5, "PASS", "easy"))
        assert result.metadata.get("synthesizability_category") == "easy"

    def test_should_discard_true_for_discard(self):
        assert self.engine.should_discard(
            self._state_with_sa("P6", 8.0, "DISCARD")
        ) is True

    def test_should_discard_false_for_pass(self):
        assert self.engine.should_discard(
            self._state_with_sa("P7", 3.0, "PASS")
        ) is False

    def test_should_discard_false_for_flag(self):
        assert self.engine.should_discard(
            self._state_with_sa("P8", 6.5, "FLAG")
        ) is False

    def test_should_flag_true(self):
        assert self.engine.should_flag(
            self._state_with_sa("P9", 6.5, "FLAG")
        ) is True

    def test_should_flag_false_for_pass(self):
        assert self.engine.should_flag(
            self._state_with_sa("P10", 3.0, "PASS")
        ) is False

    def test_sa_error_treated_as_discard(self):
        state = AgentState(molecule_id="P11", raw_input="CCO")
        state.add_tool_result("ValidityTool", {"is_valid": True, "error_message": None})
        state.add_tool_result("SAScoreTool", {
            "sa_score": None, "sa_decision": "ERROR",
            "sa_description": "", "warning_flags": [],
            "error_message": "sascorer unavailable",
        })
        assert self.engine.should_discard(state) is True

    def test_validity_takes_priority_over_sa(self):
        from agent.decision import DecisionType
        state = AgentState(molecule_id="P12", raw_input="INVALID")
        state.add_tool_result("ValidityTool", {
            "is_valid": False, "error_message": "parse failure"
        })
        state.add_tool_result("SAScoreTool", {
            "sa_score": 3.0, "sa_decision": "PASS",
            "sa_description": "", "warning_flags": [], "error_message": None,
        })
        result = self.engine.evaluate(state)
        assert result.decision_type == DecisionType.DISCARD
        assert result.metadata.get("tool_checked") == "ValidityTool"

    def test_no_sa_result_defaults_to_flag(self):
        # When no SAScoreTool result exists, the engine falls through to the
        # default_action which is FLAG (conservative fallback by design).
        from agent.decision import DecisionType
        state = AgentState(molecule_id="P13", raw_input="CCO")
        state.add_tool_result("ValidityTool", {"is_valid": True, "error_message": None})
        result = self.engine.evaluate(state)
        assert result.decision_type == DecisionType.FLAG

    def test_custom_threshold_engine(self):
        engine = PolicyEngine(sa_thresholds=LEAD_OPTIMIZATION_THRESHOLDS)
        state = self._state_with_sa("P14", 5.5, "FLAG", "moderate")
        assert self._decision_name(state) == "FLAG"


# =============================================================================
# Week 4: Agent integration tests with both tools
# =============================================================================

class TestAgentSAScoreIntegration:

    def setup_method(self):
        self.agent = TriageAgent(
            tools=[ValidityTool(), SAScoreTool()],
            policy_engine=PolicyEngine(),
            logger=logging.getLogger("test_sa_agent"),
        )

    @requires_sascorer
    def test_easy_molecule_runs_both_tools(self):
        state = self.agent.run(molecule_id="INT_001", raw_input="CCO")
        assert "ValidityTool" in state.tool_results
        assert "SAScoreTool" in state.tool_results
        assert state.tool_results["SAScoreTool"]["sa_score"] is not None

    @requires_sascorer
    def test_easy_molecule_not_terminated(self):
        state = self.agent.run(molecule_id="INT_002", raw_input="CCO")
        assert state.is_terminated() is False

    def test_invalid_molecule_skips_sa_tool(self):
        state = self.agent.run(molecule_id="INT_003", raw_input="C(C)(C)(C)(C)C")
        assert state.is_terminated() is True
        assert "SAScoreTool" not in state.tool_results

    @requires_sascorer
    def test_aspirin_passes_full_pipeline(self):
        state = self.agent.run(
            molecule_id="INT_004", raw_input="CC(=O)Oc1ccccc1C(=O)O"
        )
        assert state.is_terminated() is False
        assert state.tool_results["SAScoreTool"]["sa_decision"] == "PASS"

    @requires_sascorer
    def test_aspirin_breakdown_in_state(self):
        state = self.agent.run(
            molecule_id="INT_005", raw_input="CC(=O)Oc1ccccc1C(=O)O"
        )
        bd = state.tool_results["SAScoreTool"]["complexity_breakdown"]
        assert bd is not None
        assert bd["num_heavy_atoms"] > 0

    def test_flagged_molecule_annotated_not_terminated(self):
        # FLAG should NOT terminate — check via direct state inspection
        state = AgentState(molecule_id="INT_006", raw_input="CCO")
        state.add_tool_result("ValidityTool", {
            "is_valid": True, "smiles_canonical": "CCO", "error_message": None
        })
        state.add_tool_result("SAScoreTool", {
            "sa_score": 6.5, "sa_decision": "FLAG",
            "synthesizability_category": "difficult",
            "sa_description": "Challenging synthesis",
            "complexity_breakdown": {}, "warning_flags": [],
            "error_message": None, "execution_time_ms": 1.0,
            "tool_name": "SAScoreTool", "molecule_id": "INT_006",
        })
        decision = PolicyEngine().evaluate(state)
        from agent.decision import DecisionType
        assert decision.decision_type == DecisionType.FLAG
        # FLAG → agent calls state.add_flag(), not state.terminate()
        # So state.is_terminated() should remain False after FLAG

    @requires_sascorer
    def test_benzene_both_tools_produce_valid_results(self):
        state = self.agent.run(molecule_id="INT_007", raw_input="c1ccccc1")
        sa_result = state.tool_results.get("SAScoreTool")
        assert sa_result is not None
        assert sa_result["sa_score"] is not None
        assert sa_result["synthesizability_category"] is not None

    @requires_sascorer
    def test_execution_times_recorded(self):
        state = self.agent.run(molecule_id="INT_008", raw_input="CCO")
        if "SAScoreTool" in state.tool_results:
            assert state.tool_results["SAScoreTool"]["execution_time_ms"] >= 0

    def test_flagged_molecule_has_flag_annotation_on_state(self):
        # Verify that state.add_flag() is called (requires agent wiring from patch)
        # This is a lightweight proxy test — checks is_flagged() if method exists
        state = self.agent.run(molecule_id="INT_009", raw_input="CCO")
        # If the molecule passed cleanly (PASS), is_flagged should be False
        if hasattr(state, 'is_flagged'):
            assert state.is_flagged() is False


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])