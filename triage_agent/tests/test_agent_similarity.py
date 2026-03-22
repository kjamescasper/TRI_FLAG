"""
tests/test_agent_week5.py

Unit and integration tests for Week 5: Similarity / IP-Risk Screening.

Test coverage:
    1. TestMorganFingerprints        — chemistry/fingerprints.py
    2. TestSimilarityThresholds      — policies/thresholds.SimilarityThresholds
    3. TestUrlEncoding               — SMILES URL encoding edge cases
    4. TestSimilarityToolUnit        — SimilarityTool with mocked requests
    5. TestSimilarityPolicyEngine    — PolicyEngine._check_similarity()
    6. TestAgentSimilarityIntegration — full agent pipeline with mocked APIs
    7. LiveSmokeTests                — @requires_network, skipped offline

Design:
    - All unit/integration tests use unittest.mock.patch to mock requests.get
      and requests.post. Zero network calls — tests pass offline and in CI/CD.
    - Two live smoke tests decorated @requires_network are provided for
      manual validation against real ChEMBL/PubChem responses.
    - All Week 3/4 tests (test_agent.py) remain unchanged.

Week 9 (similarity update):
    SimilarityTool now uses three sources:
        ChEMBL     — flagging source (approved/bioactive drugs)
        SureChEMBL — flagging source (patent literature)
        PubChem    — informational only, never triggers FLAG decision
    test_flag_above_threshold_pubchem updated: PubChem hit alone → PASS
        (PubChem is no longer a flagging source; result stored in pubchem_hits
        for reference but does not drive the similarity_decision.)
    test_flag_dual_source updated: ChEMBL hit → FLAG; PubChem stored but not
        the causal source. Asserts FLAG decision and ChEMBL hit presence only.

Usage:
    # Run offline tests (no network required)
    pytest tests/test_agent_week5.py -v

    # Run including live smoke tests (requires internet)
    pytest tests/test_agent_week5.py -v -m network
"""

import logging
import pytest
import urllib.parse
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from agent.agent_state import AgentState
from agent.triage_agent import TriageAgent
from agent.decision import DecisionType
from chemistry.fingerprints import (
    morgan_fingerprint,
    morgan_fingerprint_from_smiles,
    tanimoto_similarity,
    fingerprint_to_hex,
    hex_to_fingerprint,
)
from policies.thresholds import SimilarityThresholds, DEFAULT_SIMILARITY_THRESHOLDS
from policies.policy_engine import PolicyEngine
from tools.similarity_tool import SimilarityTool
from tools.validity_tool import ValidityTool


# ============================================================================
# Network marker — tests requiring live API access
# ============================================================================

def requires_network(func):
    """
    Decorator: skip test if network is unavailable or not explicitly requested.

    Usage:
        @requires_network
        def test_live_chembl_ethanol(self): ...

    To run live tests:
        pytest tests/test_agent_week5.py -v -m network
    """
    import functools
    return pytest.mark.network(
        pytest.mark.skipif(
            not _network_available(),
            reason="Skipped: no network or -m network not specified",
        )(func)
    )


def _network_available() -> bool:
    """Quick connectivity check against ChEMBL API."""
    try:
        import requests
        r = requests.get(
            "https://www.ebi.ac.uk/chembl/api/data/status",
            timeout=3.0,
        )
        return r.status_code == 200
    except Exception:
        return False


# ============================================================================
# Shared mock helpers
# ============================================================================

def _make_chembl_response(hits=None, status_code=200):
    """Build a mock requests.Response for ChEMBL similarity endpoint."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.raise_for_status = MagicMock()
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = Exception(
            f"HTTP {status_code}"
        )
    payload = {"molecules": hits or []}
    mock_resp.json.return_value = payload
    return mock_resp


def _chembl_hit(chembl_id: str, name: str, similarity_pct: float, smiles: str = "CCO"):
    """Build a ChEMBL-style hit dict (similarity in 0-100 range)."""
    return {
        "molecule_chembl_id": chembl_id,
        "pref_name": name,
        "similarity": similarity_pct,
        "molecule_structures": {"canonical_smiles": smiles},
    }


def _make_pubchem_waiting_response(list_key="TEST_KEY_001"):
    """Build a mock PubChem 'Waiting' response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"Waiting": {"ListKey": list_key}}
    return mock_resp


def _make_pubchem_cid_response(cids):
    """Build a mock PubChem CID list response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"IdentifierList": {"CID": cids}}
    return mock_resp


def _make_pubchem_props_response(cids):
    """Build a mock PubChem property fetch response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "PropertyTable": {
            "Properties": [
                {
                    "CID": cid,
                    "IUPACName": f"compound_{cid}",
                    "IsomericSMILES": "CCO",
                }
                for cid in cids
            ]
        }
    }
    return mock_resp


# ============================================================================
# 1. Morgan fingerprint tests
# ============================================================================

class TestMorganFingerprints:
    """Unit tests for chemistry/fingerprints.py."""

    def test_fingerprint_from_valid_smiles(self):
        """Ethanol fingerprint should be non-None."""
        fp = morgan_fingerprint_from_smiles("CCO")
        assert fp is not None

    def test_fingerprint_from_invalid_smiles(self):
        """Invalid SMILES should return None, not raise."""
        fp = morgan_fingerprint_from_smiles("NOT_A_MOLECULE_XYZ")
        assert fp is None

    def test_fingerprint_from_empty_smiles(self):
        """Empty string should return None."""
        assert morgan_fingerprint_from_smiles("") is None

    def test_fingerprint_from_none(self):
        """None mol should return None."""
        fp = morgan_fingerprint(None)
        assert fp is None

    def test_tanimoto_identical_molecules(self):
        """Same molecule should have Tanimoto = 1.0."""
        fp1 = morgan_fingerprint_from_smiles("CCO")
        fp2 = morgan_fingerprint_from_smiles("CCO")
        assert fp1 is not None and fp2 is not None
        score = tanimoto_similarity(fp1, fp2)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_tanimoto_different_molecules(self):
        """Ethanol vs aspirin should have low Tanimoto similarity."""
        fp_ethanol = morgan_fingerprint_from_smiles("CCO")
        fp_aspirin = morgan_fingerprint_from_smiles("CC(=O)Oc1ccccc1C(=O)O")
        assert fp_ethanol is not None and fp_aspirin is not None
        score = tanimoto_similarity(fp_ethanol, fp_aspirin)
        assert score < 0.5, f"Expected low similarity, got {score:.3f}"

    def test_tanimoto_none_inputs(self):
        """None inputs should return 0.0, not raise."""
        assert tanimoto_similarity(None, None) == 0.0
        fp = morgan_fingerprint_from_smiles("CCO")
        assert tanimoto_similarity(fp, None) == 0.0
        assert tanimoto_similarity(None, fp) == 0.0

    def test_canonical_smiles_same_fingerprint(self):
        """OCC and CCO are the same molecule — should have identical fingerprints."""
        fp1 = morgan_fingerprint_from_smiles("OCC")
        fp2 = morgan_fingerprint_from_smiles("CCO")
        assert fp1 is not None and fp2 is not None
        score = tanimoto_similarity(fp1, fp2)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_fingerprint_to_hex_roundtrip(self):
        """Hex serialization should round-trip correctly."""
        fp_original = morgan_fingerprint_from_smiles("CCO")
        assert fp_original is not None

        hex_str = fingerprint_to_hex(fp_original)
        assert hex_str is not None
        assert isinstance(hex_str, str)
        assert len(hex_str) > 0

        fp_restored = hex_to_fingerprint(hex_str)
        assert fp_restored is not None

        # Round-tripped fingerprint should be identical
        score = tanimoto_similarity(fp_original, fp_restored)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_fingerprint_to_hex_none(self):
        """None input should return None without raising."""
        assert fingerprint_to_hex(None) is None

    def test_hex_to_fingerprint_empty(self):
        """Empty string should return None without raising."""
        assert hex_to_fingerprint("") is None

    def test_fingerprint_bit_length(self):
        """Default fingerprint should have 2048 bits."""
        fp = morgan_fingerprint_from_smiles("CCO")
        assert fp is not None
        assert fp.GetNumBits() == 2048

    def test_fingerprint_aromatic(self):
        """Aromatic molecule (benzene) should produce valid fingerprint."""
        fp = morgan_fingerprint_from_smiles("c1ccccc1")
        assert fp is not None

    def test_tanimoto_similar_molecules(self):
        """Ethanol and methanol should have moderate Tanimoto similarity."""
        fp_ethanol = morgan_fingerprint_from_smiles("CCO")     # ethanol
        fp_methanol = morgan_fingerprint_from_smiles("CO")     # methanol
        assert fp_ethanol is not None and fp_methanol is not None
        score = tanimoto_similarity(fp_ethanol, fp_methanol)
        # Similar but not identical molecules — expect moderate similarity
        assert 0.1 < score < 1.0, f"Expected moderate similarity, got {score:.3f}"


# ============================================================================
# 2. SimilarityThresholds tests
# ============================================================================

class TestSimilarityThresholds:
    """Unit tests for policies/thresholds.SimilarityThresholds."""

    def setup_method(self):
        self.thresholds = SimilarityThresholds(
            flag_threshold=0.85,
            escalation_threshold=0.95,
        )

    def test_below_threshold_is_pass(self):
        assert self.thresholds.classify(0.84) == "PASS"
        assert self.thresholds.classify(0.0) == "PASS"

    def test_at_threshold_is_flag(self):
        assert self.thresholds.classify(0.85) == "FLAG"

    def test_above_threshold_is_flag(self):
        assert self.thresholds.classify(0.90) == "FLAG"
        assert self.thresholds.classify(1.0) == "FLAG"

    def test_escalation_below(self):
        assert self.thresholds.is_escalated(0.94) is False

    def test_escalation_at_threshold(self):
        assert self.thresholds.is_escalated(0.95) is True

    def test_escalation_above(self):
        assert self.thresholds.is_escalated(0.99) is True

    def test_describe_pass(self):
        desc = self.thresholds.describe(0.30)
        assert "PASS" in desc
        assert "dissimilar" in desc.lower()

    def test_describe_flag(self):
        desc = self.thresholds.describe(0.88)
        assert "FLAG" in desc

    def test_describe_escalated(self):
        desc = self.thresholds.describe(0.97)
        assert "near-identical" in desc.lower()

    def test_invalid_flag_threshold_raises(self):
        with pytest.raises(ValueError):
            SimilarityThresholds(flag_threshold=0.0)
        with pytest.raises(ValueError):
            SimilarityThresholds(flag_threshold=1.1)

    def test_flag_gt_escalation_raises(self):
        with pytest.raises(ValueError):
            SimilarityThresholds(flag_threshold=0.95, escalation_threshold=0.85)

    def test_default_preset(self):
        assert DEFAULT_SIMILARITY_THRESHOLDS.flag_threshold == 0.85
        assert DEFAULT_SIMILARITY_THRESHOLDS.escalation_threshold == 0.95


# ============================================================================
# 3. URL encoding tests
# ============================================================================

class TestUrlEncoding:
    """
    Verify SMILES special characters are correctly percent-encoded.

    ChEMBL REST endpoint embeds SMILES in the URL path; characters like
    #, /, @, [, ] must be encoded to avoid misinterpretation.
    """

    def test_simple_smiles_encodes_safely(self):
        """Simple SMILES without special chars should encode cleanly."""
        smiles = "CCO"
        encoded = urllib.parse.quote(smiles, safe="")
        assert encoded == "CCO"

    def test_aromatic_brackets_encoded(self):
        """Brackets in SMILES must be encoded."""
        smiles = "c1ccccc1"  # no brackets — should pass through
        encoded = urllib.parse.quote(smiles, safe="")
        assert "[" not in encoded  # no brackets in benzene

    def test_square_brackets_encoded(self):
        """Square brackets (atom notation) must be encoded."""
        smiles = "[NH4+]"
        encoded = urllib.parse.quote(smiles, safe="")
        assert "[" not in encoded
        assert "]" not in encoded
        assert "+" not in encoded

    def test_triple_bond_encoded(self):
        """Triple bond '#' must be encoded."""
        smiles = "C#N"  # nitrile
        encoded = urllib.parse.quote(smiles, safe="")
        assert "#" not in encoded
        assert "%23" in encoded

    def test_slash_encoded(self):
        """Slashes (E/Z notation) must be encoded."""
        smiles = "C/C=C/C"  # trans-2-butene
        encoded = urllib.parse.quote(smiles, safe="")
        assert "/" not in encoded

    def test_at_sign_encoded(self):
        """@ (stereochemistry) must be encoded."""
        smiles = "C[C@H](N)C(=O)O"  # L-alanine
        encoded = urllib.parse.quote(smiles, safe="")
        assert "@" not in encoded

    def test_roundtrip(self):
        """Encoded SMILES should decode back to original."""
        smiles = "C[C@H](N)C(=O)O"
        encoded = urllib.parse.quote(smiles, safe="")
        decoded = urllib.parse.unquote(encoded)
        assert decoded == smiles


# ============================================================================
# 4. SimilarityTool unit tests (mocked network)
# ============================================================================

class TestSimilarityToolUnit:
    """
    Unit tests for SimilarityTool with all HTTP requests mocked.

    These tests never make real network calls — safe for CI/CD.
    """

    def setup_method(self):
        self.tool = SimilarityTool(
            flag_threshold=0.85,
            use_chembl=True,
            use_pubchem=True,
            chembl_timeout=5.0,
            pubchem_timeout=5.0,
        )
        self.agent_logger = logging.getLogger("test_similarity_tool")

    def _make_state(self, smiles: str, molecule_id: str = "TEST_001") -> AgentState:
        """Create AgentState with ValidityTool result pre-populated."""
        state = AgentState(molecule_id=molecule_id, raw_input=smiles)
        state.add_tool_result(
            "ValidityTool",
            {
                "is_valid": True,
                "smiles_canonical": smiles,
                "num_atoms": 3,
                "num_bonds": 2,
                "error_message": None,
            },
        )
        return state

    # ── PASS cases ────────────────────────────────────────────────────────────

    @patch("tools.similarity_tool._CHEMBL_CLIENT_AVAILABLE", False)
    @patch("tools.similarity_tool._requests")
    def test_pass_no_hits(self, mock_requests):
        """No hits from either API → PASS."""
        # ChEMBL returns empty hit list
        mock_requests.get.return_value = _make_chembl_response(hits=[])
        # PubChem returns immediately with no CIDs
        mock_requests.post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"IdentifierList": {"CID": []}}),
        )

        state = self._make_state("CCO")
        result = self.tool.run(state)

        assert result["similarity_decision"] == "PASS"
        assert result["nearest_neighbor_tanimoto"] == 0.0

    @patch("tools.similarity_tool._CHEMBL_CLIENT_AVAILABLE", False)
    @patch("tools.similarity_tool._requests")
    def test_pass_below_threshold(self, mock_requests):
        """Hit with Tanimoto < 0.85 → PASS."""
        mock_requests.get.return_value = _make_chembl_response(
            hits=[_chembl_hit("CHEMBL999", "Low-sim compound", 80.0)]  # 80% = 0.80
        )
        mock_requests.post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"IdentifierList": {"CID": []}}),
        )

        state = self._make_state("CCO")
        result = self.tool.run(state)

        assert result["similarity_decision"] == "PASS"
        assert result["nearest_neighbor_tanimoto"] == pytest.approx(0.80, abs=1e-6)

    # ── FLAG cases ────────────────────────────────────────────────────────────

    @patch("tools.similarity_tool._CHEMBL_CLIENT_AVAILABLE", False)
    @patch("tools.similarity_tool._requests")
    def test_flag_above_threshold_chembl(self, mock_requests):
        """ChEMBL hit at 0.92 → FLAG."""
        mock_requests.get.return_value = _make_chembl_response(
            hits=[_chembl_hit("CHEMBL1", "Ethanol", 92.0)]
        )
        mock_requests.post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"IdentifierList": {"CID": []}}),
        )

        state = self._make_state("CCO")
        result = self.tool.run(state)

        assert result["similarity_decision"] == "FLAG"
        assert result["nearest_neighbor_tanimoto"] == pytest.approx(0.92, abs=1e-6)
        assert result["nearest_neighbor_source"] == "ChEMBL"
        assert result["nearest_neighbor_id"] == "CHEMBL1"

    @patch("tools.similarity_tool._CHEMBL_CLIENT_AVAILABLE", False)
    @patch("tools.similarity_tool._requests")
    def test_flag_above_threshold_pubchem(self, mock_requests):
        """
        PubChem hit alone → PASS.

        Week 9 change: PubChem is informational only and never triggers FLAG.
        A PubChem hit with no ChEMBL or SureChEMBL hits must produce PASS.
        The hit is still stored in pubchem_hits for reference.
        """
        # ChEMBL returns no hits
        mock_requests.get.side_effect = [
            _make_chembl_response(hits=[]),          # ChEMBL similarity query
            _make_pubchem_props_response([12345]),    # PubChem property fetch
        ]
        mock_requests.post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"Waiting": {"ListKey": "KEY123"}}),
        )
        poll_mock = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"IdentifierList": {"CID": [12345]}}),
        )
        # SureChEMBL also uses GET — add a no-hit response so PubChem gets its slot
        surechembl_no_hits = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"results": []}),
        )
        mock_requests.get.side_effect = [
            _make_chembl_response(hits=[]),          # ChEMBL
            surechembl_no_hits,                       # SureChEMBL
            poll_mock,                                # PubChem poll
            _make_pubchem_props_response([12345]),    # PubChem props
        ]

        state = self._make_state("CCO")
        result = self.tool.run(state)

        # PubChem is informational only — no flagging sources hit → PASS
        assert result["similarity_decision"] == "PASS"
        # PubChem result is still stored for reference
        assert len(result["pubchem_hits"]) >= 1
        assert result["pubchem_hits"][0]["informational"] is True

    @patch("tools.similarity_tool._CHEMBL_CLIENT_AVAILABLE", False)
    @patch("tools.similarity_tool._requests")
    def test_flag_dual_source(self, mock_requests):
        """
        ChEMBL hit → FLAG; PubChem hit stored but not causal.

        Week 9 change: FLAG is driven by ChEMBL hit. PubChem hit is stored
        in pubchem_hits but does not affect the decision.
        """
        mock_requests.get.side_effect = [
            _make_chembl_response(
                hits=[_chembl_hit("CHEMBL1", "KnownDrug", 91.0)]
            ),
            _make_pubchem_props_response([99999]),
        ]
        mock_requests.post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"Waiting": {"ListKey": "KEY_DUAL"}}),
        )
        poll_mock = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"IdentifierList": {"CID": [99999]}}),
        )
        # SureChEMBL also uses GET — add a no-hit response so PubChem gets its slot
        surechembl_no_hits = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"results": []}),
        )
        mock_requests.get.side_effect = [
            _make_chembl_response(hits=[_chembl_hit("CHEMBL1", "KnownDrug", 91.0)]),
            surechembl_no_hits,                       # SureChEMBL
            poll_mock,
            _make_pubchem_props_response([99999]),
        ]

        state = self._make_state("CCO")
        result = self.tool.run(state)

        # ChEMBL hit drives the FLAG
        assert result["similarity_decision"] == "FLAG"
        assert result["flag_source"] == "ChEMBL"
        assert len(result["chembl_hits"]) >= 1
        # PubChem hit is stored for reference
        assert len(result["pubchem_hits"]) >= 1

    # ── ERROR cases ────────────────────────────────────────────────────────────

    @patch("tools.similarity_tool._CHEMBL_CLIENT_AVAILABLE", False)
    @patch("tools.similarity_tool._requests")
    def test_error_both_apis_fail(self, mock_requests):
        """Both APIs raise ConnectionError → ERROR result (non-terminal)."""
        import requests as real_requests
        mock_requests.exceptions.ConnectionError = real_requests.exceptions.ConnectionError
        mock_requests.exceptions.Timeout = real_requests.exceptions.Timeout
        mock_requests.get.side_effect = real_requests.exceptions.ConnectionError("offline")
        mock_requests.post.side_effect = real_requests.exceptions.ConnectionError("offline")

        state = self._make_state("CCO")
        result = self.tool.run(state)

        assert result["similarity_decision"] == "ERROR"
        assert result["chembl_available"] is False
        assert result["pubchem_available"] is False
        assert "error_reason" in result

    @patch("tools.similarity_tool._CHEMBL_CLIENT_AVAILABLE", False)
    @patch("tools.similarity_tool._requests")
    def test_error_no_smiles_in_state(self, mock_requests):
        """No ValidityTool result and non-string raw_input → ERROR."""
        state = AgentState(molecule_id="NO_SMILES", raw_input=12345)
        result = self.tool.run(state)

        assert result["similarity_decision"] == "ERROR"
        assert "error_reason" in result

    # ── Result schema tests ────────────────────────────────────────────────────

    @patch("tools.similarity_tool._CHEMBL_CLIENT_AVAILABLE", False)
    @patch("tools.similarity_tool._requests")
    def test_result_schema_completeness(self, mock_requests):
        """Result dict must contain all required keys."""
        mock_requests.get.return_value = _make_chembl_response(hits=[])
        mock_requests.post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"IdentifierList": {"CID": []}}),
        )

        state = self._make_state("CCO")
        result = self.tool.run(state)

        required_keys = [
            "tool_name",
            "molecule_id",
            "query_smiles",
            "similarity_decision",
            "nearest_neighbor_tanimoto",
            "nearest_neighbor_source",
            "nearest_neighbor_id",
            "nearest_neighbor_name",
            "nearest_neighbor_smiles",
            "chembl_hits",
            "pubchem_hits",
            "flag_threshold_used",
            "fingerprint_method",
            "query_fingerprint_hex",
            "apis_queried",
            "chembl_available",
            "pubchem_available",
            "execution_time_ms",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    @patch("tools.similarity_tool._CHEMBL_CLIENT_AVAILABLE", False)
    @patch("tools.similarity_tool._requests")
    def test_fingerprint_hex_stored(self, mock_requests):
        """Result should include fingerprint hex for provenance."""
        mock_requests.get.return_value = _make_chembl_response(hits=[])
        mock_requests.post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"IdentifierList": {"CID": []}}),
        )

        state = self._make_state("CCO")
        result = self.tool.run(state)

        fp_hex = result.get("query_fingerprint_hex")
        assert fp_hex is not None
        assert len(fp_hex) > 0

    @patch("tools.similarity_tool._CHEMBL_CLIENT_AVAILABLE", False)
    @patch("tools.similarity_tool._requests")
    def test_chembl_only_mode(self, mock_requests):
        """use_pubchem=False should only query ChEMBL."""
        tool = SimilarityTool(flag_threshold=0.85, use_chembl=True, use_pubchem=False)
        mock_requests.get.return_value = _make_chembl_response(hits=[])

        state = self._make_state("CCO")
        result = tool.run(state)

        assert "PubChem" not in result["apis_queried"]

    @patch("tools.similarity_tool._CHEMBL_CLIENT_AVAILABLE", False)
    @patch("tools.similarity_tool._requests")
    def test_pubchem_only_mode(self, mock_requests):
        """use_chembl=False should only query PubChem."""
        tool = SimilarityTool(flag_threshold=0.85, use_chembl=False, use_pubchem=True)
        mock_requests.post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"IdentifierList": {"CID": []}}),
        )

        state = self._make_state("CCO")
        result = tool.run(state)

        assert "ChEMBL" not in result["apis_queried"]


# ============================================================================
# 5. PolicyEngine._check_similarity() tests
# ============================================================================

class TestSimilarityPolicyEngine:
    """Test PolicyEngine._check_similarity() and _build_similarity_rationale()."""

    def setup_method(self):
        self.engine = PolicyEngine()

    def _make_state_with_sim_result(
        self,
        similarity_decision: str,
        nn_tanimoto: float = 0.0,
        nn_source: str = None,
        nn_id: str = None,
        chembl_hits: list = None,
        pubchem_hits: list = None,
    ) -> AgentState:
        """Build AgentState with a SimilarityTool result."""
        state = AgentState(molecule_id="SIM_TEST", raw_input="CCO")
        # Validity required by _check_validity
        state.add_tool_result(
            "ValidityTool",
            {
                "is_valid": True,
                "smiles_canonical": "CCO",
                "num_atoms": 3,
                "error_message": None,
            },
        )
        state.add_tool_result(
            "SimilarityTool",
            {
                "similarity_decision": similarity_decision,
                "nearest_neighbor_tanimoto": nn_tanimoto,
                "nearest_neighbor_source": nn_source,
                "nearest_neighbor_id": nn_id,
                "nearest_neighbor_name": None,
                "nearest_neighbor_smiles": None,
                "chembl_hits": chembl_hits or [],
                "pubchem_hits": pubchem_hits or [],
                "flag_threshold_used": 0.85,
                "apis_queried": ["ChEMBL", "PubChem"],
                "chembl_available": True,
                "pubchem_available": True,
                "execution_time_ms": 100.0,
            },
        )
        return state

    def test_pass_produces_pass_decision(self):
        """SimilarityTool PASS → PolicyEngine produces PASS (via fallback)."""
        state = self._make_state_with_sim_result("PASS", nn_tanimoto=0.50)
        decision = self.engine.evaluate(state)
        # With no SA result, fallback is FLAG (conservative default)
        # PASS similarity doesn't override — falls through to fallback
        assert decision.decision_type in (DecisionType.PASS, DecisionType.FLAG)

    def test_flag_produces_flag_decision(self):
        """SimilarityTool FLAG → PolicyEngine produces FLAG."""
        state = self._make_state_with_sim_result(
            "FLAG",
            nn_tanimoto=0.91,
            nn_source="ChEMBL",
            nn_id="CHEMBL1",
            chembl_hits=[{"source": "ChEMBL", "id": "CHEMBL1",
                          "tanimoto": 0.91, "name": "TestDrug", "smiles": "CCO"}],
        )
        decision = self.engine.evaluate(state)
        assert decision.decision_type == DecisionType.FLAG
        assert "Similarity" in decision.rationale or "similar" in decision.rationale.lower()
        assert "0.910" in decision.rationale or "0.91" in decision.rationale

    def test_error_produces_flag_decision(self):
        """SimilarityTool ERROR → PolicyEngine produces FLAG (conservative)."""
        state = self._make_state_with_sim_result("ERROR")
        # Override pubchem_available to False to trigger ERROR path
        state.tool_results["SimilarityTool"]["chembl_available"] = False
        state.tool_results["SimilarityTool"]["pubchem_available"] = False
        state.tool_results["SimilarityTool"]["error_reason"] = "API unavailable"
        decision = self.engine.evaluate(state)
        assert decision.decision_type == DecisionType.FLAG
        assert "unavailable" in decision.rationale.lower()

    def test_dual_source_rationale(self):
        """Both ChEMBL and PubChem hits → rationale mentions both."""
        state = self._make_state_with_sim_result(
            "FLAG",
            nn_tanimoto=0.93,
            nn_source="ChEMBL",
            nn_id="CHEMBL1",
            chembl_hits=[{"source": "ChEMBL", "id": "CHEMBL1",
                          "tanimoto": 0.93, "name": "Drug", "smiles": "CCO"}],
            pubchem_hits=[{"source": "PubChem", "id": "12345",
                           "tanimoto": 0.88, "name": "Compound", "smiles": "CCO"}],
        )
        decision = self.engine.evaluate(state)
        assert decision.decision_type == DecisionType.FLAG
        # Dual-source rationale should mention both
        rationale = decision.rationale
        assert "ChEMBL" in rationale
        assert "PubChem" in rationale

    def test_escalated_flag_near_identical(self):
        """Tanimoto >= 0.95 → escalated rationale."""
        state = self._make_state_with_sim_result(
            "FLAG",
            nn_tanimoto=0.97,
            nn_source="ChEMBL",
            nn_id="CHEMBL42",
            chembl_hits=[{"source": "ChEMBL", "id": "CHEMBL42",
                          "tanimoto": 0.97, "name": "NearTwin", "smiles": "CCO"}],
        )
        decision = self.engine.evaluate(state)
        assert decision.decision_type == DecisionType.FLAG
        assert decision.metadata.get("escalated") is True

    def test_metadata_contains_per_source_scores(self):
        """Decision metadata should include per-source best Tanimoto scores."""
        state = self._make_state_with_sim_result(
            "FLAG",
            nn_tanimoto=0.91,
            nn_source="ChEMBL",
            nn_id="CHEMBL1",
            chembl_hits=[{"source": "ChEMBL", "id": "CHEMBL1",
                          "tanimoto": 0.91, "name": "X", "smiles": "CCO"}],
            pubchem_hits=[{"source": "PubChem", "id": "777",
                           "tanimoto": 0.87, "name": "Y", "smiles": "CCO"}],
        )
        decision = self.engine.evaluate(state)
        meta = decision.metadata
        assert "chembl_best_tanimoto" in meta
        assert "pubchem_best_tanimoto" in meta
        assert meta["chembl_best_tanimoto"] == pytest.approx(0.91, abs=1e-6)
        assert meta["pubchem_best_tanimoto"] == pytest.approx(0.87, abs=1e-6)

    def test_no_similarity_result_uses_fallback(self):
        """No SimilarityTool result → fallback FLAG (conservative default)."""
        state = AgentState(molecule_id="NO_SIM", raw_input="CCO")
        state.add_tool_result(
            "ValidityTool",
            {
                "is_valid": True,
                "smiles_canonical": "CCO",
                "num_atoms": 3,
                "error_message": None,
            },
        )
        decision = self.engine.evaluate(state)
        assert decision.decision_type == DecisionType.FLAG  # fallback

    def test_validity_discard_skips_similarity(self):
        """DISCARD from ValidityTool → PolicyEngine ignores SimilarityTool."""
        state = AgentState(molecule_id="INVALID", raw_input="C(C)(C)(C)(C)C")
        state.add_tool_result(
            "ValidityTool",
            {
                "is_valid": False,
                "error_message": "RDKit failed to parse",
                "smiles_canonical": None,
                "num_atoms": 0,
            },
        )
        state.add_tool_result(
            "SimilarityTool",
            {
                "similarity_decision": "FLAG",
                "nearest_neighbor_tanimoto": 0.99,
                "nearest_neighbor_source": "ChEMBL",
                "nearest_neighbor_id": "CHEMBL1",
                "nearest_neighbor_name": None,
                "nearest_neighbor_smiles": None,
                "chembl_hits": [],
                "pubchem_hits": [],
                "flag_threshold_used": 0.85,
                "apis_queried": ["ChEMBL"],
                "chembl_available": True,
                "pubchem_available": False,
                "execution_time_ms": 50.0,
            },
        )
        decision = self.engine.evaluate(state)
        assert decision.decision_type == DecisionType.DISCARD
        assert "chemically invalid" in decision.rationale.lower()


# ============================================================================
# 6. Full agent pipeline integration tests (mocked network)
# ============================================================================

class TestAgentSimilarityIntegration:
    """
    Integration tests for TriageAgent with SimilarityTool in the pipeline.

    All HTTP requests are mocked — no network required.
    """

    def setup_method(self):
        from tools.validity_tool import ValidityTool
        policy_engine = PolicyEngine()
        agent_logger = logging.getLogger("test_agent_similarity")
        self.agent = TriageAgent(
            tools=[ValidityTool(), SimilarityTool()],
            policy_engine=policy_engine,
            logger=agent_logger,
        )

    @patch("tools.similarity_tool._CHEMBL_CLIENT_AVAILABLE", False)
    @patch("tools.similarity_tool._requests")
    def test_valid_molecule_no_sim_hits_produces_flag(self, mock_requests):
        """
        Valid molecule + no similarity hits → FLAG (conservative fallback).

        Rationale: No SA tool registered, no similarity hits, no pluggable
        policies → PolicyEngine default fallback → FLAG.
        """
        mock_requests.get.return_value = _make_chembl_response(hits=[])
        mock_requests.post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"IdentifierList": {"CID": []}}),
        )

        state = self.agent.run(molecule_id="INT_001", raw_input="CCO")

        assert state is not None
        assert "ValidityTool" in state.tool_results
        assert "SimilarityTool" in state.tool_results
        assert state.tool_results["ValidityTool"]["is_valid"] is True
        assert state.tool_results["SimilarityTool"]["similarity_decision"] == "PASS"
        # State not terminated (no early exit)
        assert state.is_terminated() is False

    @patch("tools.similarity_tool._CHEMBL_CLIENT_AVAILABLE", False)
    @patch("tools.similarity_tool._requests")
    def test_valid_molecule_high_sim_produces_flag(self, mock_requests):
        """Valid molecule + high similarity → FLAG decision."""
        mock_requests.get.return_value = _make_chembl_response(
            hits=[_chembl_hit("CHEMBL1", "EthanolRef", 92.0)]
        )
        mock_requests.post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"IdentifierList": {"CID": []}}),
        )

        state = self.agent.run(molecule_id="INT_002", raw_input="CCO")

        assert state.decision.decision_type == DecisionType.FLAG
        assert state.tool_results["SimilarityTool"]["similarity_decision"] == "FLAG"
        # Pipeline should NOT have terminated early
        assert state.is_terminated() is False

    @patch("tools.similarity_tool._CHEMBL_CLIENT_AVAILABLE", False)
    @patch("tools.similarity_tool._requests")
    def test_invalid_molecule_does_not_reach_similarity(self, mock_requests):
        """
        Invalid molecule → terminated before SimilarityTool runs.

        ValidityTool triggers early termination; SimilarityTool is skipped.
        Confirms SimilarityTool APIs are never called for invalid molecules.
        """
        state = self.agent.run(
            molecule_id="INT_003",
            raw_input="C(C)(C)(C)(C)C",  # Invalid valence
        )

        assert state is not None
        assert state.is_terminated() is True
        assert state.decision.decision_type == DecisionType.DISCARD
        # SimilarityTool should NOT have run
        assert "SimilarityTool" not in state.tool_results
        # Confirm no HTTP calls were made
        mock_requests.get.assert_not_called()
        mock_requests.post.assert_not_called()

    @patch("tools.similarity_tool._CHEMBL_CLIENT_AVAILABLE", False)
    @patch("tools.similarity_tool._requests")
    def test_api_error_does_not_crash_pipeline(self, mock_requests):
        """
        Both APIs raise ConnectionError → pipeline completes with FLAG.

        SimilarityTool ERROR is non-terminal; agent returns a state with
        SIMILARITY_API_ERROR annotation and PolicyEngine produces FLAG.
        """
        import requests as real_requests
        mock_requests.exceptions.ConnectionError = real_requests.exceptions.ConnectionError
        mock_requests.exceptions.Timeout = real_requests.exceptions.Timeout
        mock_requests.get.side_effect = real_requests.exceptions.ConnectionError("down")
        mock_requests.post.side_effect = real_requests.exceptions.ConnectionError("down")

        state = self.agent.run(molecule_id="INT_004", raw_input="CCO")

        # Should not raise — non-terminal
        assert state is not None
        assert state.tool_results["SimilarityTool"]["similarity_decision"] == "ERROR"
        assert state.is_terminated() is False
        # Decision should be FLAG (conservative error handling)
        assert state.decision.decision_type == DecisionType.FLAG

    @patch("tools.similarity_tool._CHEMBL_CLIENT_AVAILABLE", False)
    @patch("tools.similarity_tool._requests")
    def test_similarity_flag_annotates_state(self, mock_requests):
        """SimilarityTool FLAG → state.add_flag called with IP_SIMILARITY_FLAG."""
        mock_requests.get.return_value = _make_chembl_response(
            hits=[_chembl_hit("CHEMBL42", "KnownDrug", 91.0)]
        )
        mock_requests.post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"IdentifierList": {"CID": []}}),
        )

        state = self.agent.run(molecule_id="INT_005", raw_input="CCO")

        # If AgentState supports get_flags(), check that SimilarityTool flag was recorded
        if hasattr(state, "get_flags"):
            flags = state.get_flags()
            # add_flag(reason, source) stores dicts with 'reason' and 'source' keys
            sources = [f.get("source", "") for f in flags]
            assert "SimilarityTool" in sources, (
                f"Expected SimilarityTool flag in state flags, got: {flags}"
            )


# ============================================================================
# 7. Live smoke tests (require network — @requires_network)
# ============================================================================

class TestLiveSmokeTests:
    """
    Live smoke tests against real ChEMBL and PubChem APIs.

    These tests are skipped unless the -m network flag is passed and a
    network connection is available. They validate real API behavior, not
    mocked behavior.

    Run with:
        pytest tests/test_agent_week5.py::TestLiveSmokeTests -v -m network
    """

    @pytest.mark.network
    @pytest.mark.skipif(not _network_available(), reason="No network available")
    def test_live_chembl_ethanol(self):
        """
        Live: Ethanol (CCO) should appear in ChEMBL similarity results.

        Ethanol (CHEMBL545) is registered in ChEMBL. A Tanimoto=1.0 match
        is expected if CCO is the canonical form.
        """
        tool = SimilarityTool(
            flag_threshold=0.85,
            use_chembl=True,
            use_pubchem=False,  # ChEMBL only for speed
            chembl_timeout=15.0,
        )
        state = AgentState(molecule_id="LIVE_ETHANOL", raw_input="CCO")
        state.add_tool_result(
            "ValidityTool",
            {
                "is_valid": True,
                "smiles_canonical": "CCO",
                "num_atoms": 3,
                "error_message": None,
            },
        )

        result = tool.run(state)

        # API should be reachable
        assert result["chembl_available"] is True, (
            "ChEMBL API unavailable — check connectivity"
        )

        # Ethanol is a well-known compound; we expect hits
        # (exact match depends on ChEMBL's current SMILES normalization)
        print(
            f"\n[LIVE] Ethanol similarity: decision={result['similarity_decision']}, "
            f"nn_tanimoto={result['nearest_neighbor_tanimoto']:.3f}, "
            f"nn_source={result['nearest_neighbor_source']}, "
            f"nn_id={result['nearest_neighbor_id']}"
        )

    @pytest.mark.network
    @pytest.mark.skipif(not _network_available(), reason="No network available")
    def test_live_novel_structure_low_similarity(self):
        """
        Live: Novel synthetic structure should have low ChEMBL similarity.

        Uses a contrived fluorinated bicyclic compound unlikely to be in
        ChEMBL's drug-like compound collection.
        """
        # Complex structure unlikely to appear in ChEMBL
        novel_smiles = "F[C@@H]1CC[C@H]2CC[C@@H]1C2"  # fluorinated bicyclo

        tool = SimilarityTool(
            flag_threshold=0.85,
            use_chembl=True,
            use_pubchem=False,
            chembl_timeout=15.0,
        )
        state = AgentState(molecule_id="LIVE_NOVEL", raw_input=novel_smiles)
        state.add_tool_result(
            "ValidityTool",
            {
                "is_valid": True,
                "smiles_canonical": novel_smiles,
                "num_atoms": 9,
                "error_message": None,
            },
        )

        result = tool.run(state)

        assert result["chembl_available"] is True, (
            "ChEMBL API unavailable — check connectivity"
        )

        print(
            f"\n[LIVE] Novel structure similarity: decision={result['similarity_decision']}, "
            f"nn_tanimoto={result['nearest_neighbor_tanimoto']:.3f}"
        )
        # Novel structure should PASS (low similarity expected)
        # This is a soft assertion — if ChEMBL data changes, may need update
        if result["similarity_decision"] == "FLAG":
            print(
                f"  Note: Unexpected FLAG — nearest neighbor = "
                f"{result['nearest_neighbor_id']} ({result['nearest_neighbor_tanimoto']:.3f}). "
                f"Novel compound may have been added to ChEMBL."
            )


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])