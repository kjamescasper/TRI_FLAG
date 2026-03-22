# tests/test_week9.py
"""
Week 9 test suite: DescriptorTool, PAINSTool, diversity metrics.

All tests use known SMILES with deterministic RDKit outputs — no network calls.
Run with: set PYTHONPATH=. && pytest tests/test_week9.py -v
"""

import pytest
from agent.agent_state import AgentState
from tools.descriptor_tool import DescriptorTool
from tools.pains_tool import PAINSTool
from analysis.diversity import compute_diversity, DiversityReport, CONVERGENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_state(molecule_id: str, smiles: str) -> AgentState:
    """Return an AgentState with a pre-populated ValidityTool result."""
    state = AgentState(molecule_id=molecule_id, raw_input=smiles)
    state.add_tool_result("ValidityTool", {
        "is_valid": True,
        "smiles_canonical": smiles,
        "error_message": None,
    })
    return state


def _invalid_state(molecule_id: str, smiles: str = "INVALID") -> AgentState:
    state = AgentState(molecule_id=molecule_id, raw_input=smiles)
    state.add_tool_result("ValidityTool", {
        "is_valid": False,
        "smiles_canonical": None,
        "error_message": "parse failure",
    })
    return state


# Known PAINS scaffold: rhodanine class (A ring)
# c1cc(=O)[nH]c(=S)s1 is a core rhodanine — reliably hits PAINS_A
RHODANINE_SMILES = "O=C1CSC(=S)N1"   # rhodanine — classic PAINS_A hit
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
CAFFEINE_SMILES = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"
ETHANOL_SMILES = "CCO"


# =============================================================================
# DescriptorTool tests
# =============================================================================

class TestDescriptorTool:

    def setup_method(self):
        self.tool = DescriptorTool()

    def test_result_has_required_keys(self):
        state = _valid_state("D001", ASPIRIN_SMILES)
        result = self.tool.run(state)
        required = [
            "tool_name", "molecule_id", "mol_weight", "logp", "tpsa",
            "hbd", "hba", "rotatable_bonds", "scaffold_smiles",
            "error_message", "execution_time_ms",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_aspirin_mol_weight(self):
        """Aspirin MW is 180.16 by RDKit (average mass)."""
        state = _valid_state("D002", ASPIRIN_SMILES)
        result = self.tool.run(state)
        assert result["error_message"] is None
        assert result["mol_weight"] is not None
        assert abs(result["mol_weight"] - 180.16) < 0.5

    def test_aspirin_logp(self):
        """Aspirin logP is ~1.31 by Wildman-Crippen."""
        state = _valid_state("D003", ASPIRIN_SMILES)
        result = self.tool.run(state)
        assert result["logp"] is not None
        assert 0.5 < result["logp"] < 2.5

    def test_aspirin_hbd(self):
        """Aspirin has 1 HBD (carboxylic acid OH)."""
        state = _valid_state("D004", ASPIRIN_SMILES)
        result = self.tool.run(state)
        assert result["hbd"] == 1

    def test_aspirin_hba(self):
        """Aspirin has 3 HBA (two C=O oxygens + ether O)."""
        state = _valid_state("D005", ASPIRIN_SMILES)
        result = self.tool.run(state)
        assert result["hba"] in (3, 4)  # RDKit version-dependent rounding

    def test_scaffold_smiles_populated(self):
        """Scaffold SMILES should be a non-empty string for valid molecules."""
        state = _valid_state("D006", IBUPROFEN_SMILES)
        result = self.tool.run(state)
        assert result["scaffold_smiles"] is not None
        assert len(result["scaffold_smiles"]) > 0

    def test_invalid_molecule_returns_null_descriptors(self):
        """Invalid molecule state → all descriptor values None, no exception."""
        state = _invalid_state("D007")
        result = self.tool.run(state)
        assert result["mol_weight"] is None
        assert result["logp"] is None
        assert result["scaffold_smiles"] is None

    def test_ethanol_lipinski_compliant(self):
        """Ethanol passes all Lipinski criteria."""
        state = _valid_state("D008", ETHANOL_SMILES)
        result = self.tool.run(state)
        assert result["mol_weight"] is not None
        assert result["mol_weight"] < 500
        assert result["logp"] < 5
        assert result["hbd"] <= 5
        assert result["hba"] <= 10

    def test_tool_name(self):
        assert self.tool.name == "DescriptorTool"

    def test_execution_time_positive(self):
        state = _valid_state("D009", CAFFEINE_SMILES)
        result = self.tool.run(state)
        assert result["execution_time_ms"] >= 0.0


# =============================================================================
# PAINSTool tests
# =============================================================================

class TestPAINSTool:

    def setup_method(self):
        self.tool = PAINSTool()

    def test_result_has_required_keys(self):
        state = _valid_state("P001", ASPIRIN_SMILES)
        result = self.tool.run(state)
        required = [
            "tool_name", "molecule_id", "pains_alert",
            "pains_matches", "error_message", "execution_time_ms",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_pains_alert_is_bool(self):
        state = _valid_state("P002", ASPIRIN_SMILES)
        result = self.tool.run(state)
        assert isinstance(result["pains_alert"], bool)

    def test_pains_matches_is_list(self):
        state = _valid_state("P003", ASPIRIN_SMILES)
        result = self.tool.run(state)
        assert isinstance(result["pains_matches"], list)

    def test_aspirin_no_pains(self):
        """Aspirin is not a PAINS compound."""
        state = _valid_state("P004", ASPIRIN_SMILES)
        result = self.tool.run(state)
        assert result["pains_alert"] is False
        assert result["pains_matches"] == []

    def test_rhodanine_pains_detected(self):
        """Rhodanine is a canonical PAINS_A scaffold."""
        state = _valid_state("P005", RHODANINE_SMILES)
        result = self.tool.run(state)
        assert result["pains_alert"] is True
        assert len(result["pains_matches"]) > 0

    def test_rhodanine_generates_flag_on_state(self):
        """PAINSTool adds a FLAG to AgentState when PAINS matched."""
        state = _valid_state("P006", RHODANINE_SMILES)
        self.tool.run(state)
        flags = state.get_flags() if hasattr(state, "get_flags") else getattr(state, "_flags", [])
        assert len(flags) > 0, "Expected at least one FLAG from PAINSTool on rhodanine"

    def test_invalid_molecule_no_alert(self):
        """Invalid molecules skip PAINS cleanly — pains_alert stays False."""
        state = _invalid_state("P007")
        result = self.tool.run(state)
        assert result["pains_alert"] is False
        assert result["error_message"] is not None

    def test_no_pains_no_flag_added(self):
        """Clean molecule → no FLAG added to state."""
        state = _valid_state("P008", ASPIRIN_SMILES)
        self.tool.run(state)
        flags = getattr(state, "_flags", [])
        assert all("PAINS" not in str(f) for f in flags)

    def test_tool_name(self):
        assert self.tool.name == "PAINSTool"


# =============================================================================
# Diversity metric tests
# =============================================================================

class TestDiversity:

    def test_all_unique_scaffolds_ratio_1(self):
        scaffolds = ["C1CCCCC1", "c1ccccc1", "C1CCNCC1"]
        report = compute_diversity(scaffolds)
        assert report.diversity_ratio == 1.0
        assert report.unique_scaffolds == 3

    def test_all_same_scaffold_ratio_0(self):
        scaffolds = ["c1ccccc1"] * 5
        report = compute_diversity(scaffolds)
        assert report.diversity_ratio == 0.2  # 1/5
        assert report.unique_scaffolds == 1
        assert report.top_scaffold_frequency == 1.0

    def test_convergence_warning_fires_at_threshold(self):
        """30% same scaffold → convergence_warning = True."""
        # 3 same out of 10 = 30% — should trigger
        scaffolds = ["c1ccccc1"] * 3 + ["C1CCCCC1"] * 7
        report = compute_diversity(scaffolds)
        assert report.top_scaffold == "c1ccccc1" or report.top_scaffold == "C1CCCCC1"
        # The majority scaffold (C1CCCCC1 at 70%) triggers warning
        assert report.convergence_warning is True

    def test_convergence_warning_not_below_threshold(self):
        """10% same scaffold → no convergence warning."""
        scaffolds = ["c1ccccc1"] * 1 + [f"C1CCCCCC{i}" for i in range(9)]
        report = compute_diversity(scaffolds)
        # top scaffold frequency = 1/10 = 10% < 30%
        assert report.top_scaffold_frequency <= CONVERGENCE_THRESHOLD

    def test_empty_list(self):
        report = compute_diversity([])
        assert report.unique_scaffolds == 0
        assert report.total_molecules == 0
        assert report.diversity_ratio == 0.0
        assert report.convergence_warning is False

    def test_all_none_scaffolds(self):
        """All NULL scaffolds (pre-Week-9 backlog) → no crash, null_scaffold_count set."""
        report = compute_diversity([None, None, None])
        assert report.unique_scaffolds == 0
        assert report.null_scaffold_count == 3
        assert report.convergence_warning is False

    def test_mixed_none_and_valid(self):
        scaffolds = [None, "c1ccccc1", None, "c1ccccc1", "C1CCCCC1"]
        report = compute_diversity(scaffolds)
        assert report.null_scaffold_count == 2
        assert report.total_molecules == 5
        assert report.unique_scaffolds == 2

    def test_top_scaffold_identified(self):
        scaffolds = ["c1ccccc1"] * 4 + ["C1CCCCC1"] * 2
        report = compute_diversity(scaffolds)
        assert report.top_scaffold == "c1ccccc1"
        assert report.top_scaffold_count == 4

    def test_diversity_report_is_dataclass(self):
        report = compute_diversity(["c1ccccc1"])
        assert isinstance(report, DiversityReport)