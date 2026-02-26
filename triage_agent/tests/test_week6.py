"""
tests/test_week6.py

TRI_FLAG Week 6 — Tests for RationaleBuilder and RunRecord

Coverage:
    1. TestRationaleBuilderValidity    — validity pass / discard / skipped
    2. TestRationaleBuilderSAScore     — SA pass / flag / discard / error
    3. TestRationaleBuilderSimilarity  — similarity pass / flag / escalated / error / skipped
    4. TestRationaleBuilderSummary     — summary string composition
    5. TestFormatText                  — text rendering contract
    6. TestRunRecordBuilder            — field mapping from state + explanation
    7. TestRunRecordPersistence        — save / load_all / load_as_dicts
    8. TestRunRecordRoundTrip          — to_dict / from_dict symmetry
    9. TestEarlyTermination            — partial-run (validity DISCARD) path

All tests are offline — zero network calls, zero RDKit dependency.
State and tool results are constructed from plain dicts.

Usage:
    pytest tests/test_week6.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from reporting.rationale_builder import (
    ExplanationSection,
    RationaleBuilder,
    TriageExplanation,
    format_dict,
    format_text,
)
from reporting.run_record import (
    AIReviewRecord,
    RunRecord,
    RunRecordBuilder,
    load_all,
    load_as_dicts,
    save,
)


# ============================================================================
# Shared mock helpers
# ============================================================================

def _make_state(
    molecule_id: str = "mol_001",
    raw_input: str = "CCO",
    decision_type_name: str = "PASS",
    decision_rationale: str = "All checks passed.",
    tool_results: Optional[Dict[str, Any]] = None,
    flags: Optional[list] = None,
    terminated: bool = False,
    termination_reason: Optional[str] = None,
) -> MagicMock:
    """Build a minimal mock AgentState with configurable tool results."""
    state = MagicMock()
    state.molecule_id = molecule_id
    state.raw_input = raw_input

    # Decision mock
    dt = MagicMock()
    dt.name = decision_type_name
    decision = MagicMock()
    decision.decision_type = dt
    decision.rationale = decision_rationale
    state.decision = decision

    state.tool_results = tool_results or {}
    state.get_flags.return_value = flags or []
    state.is_terminated.return_value = terminated
    state.get_termination_reason.return_value = termination_reason
    state.execution_start_time = None  # skips timing in RunRecordBuilder

    return state


def _validity_pass(smiles: str = "CCO", num_atoms: int = 3, num_bonds: int = 2) -> Dict:
    return {
        "tool_name": "ValidityTool",
        "molecule_id": "mol_001",
        "is_valid": True,
        "error_message": None,
        "smiles_canonical": smiles,
        "num_atoms": num_atoms,
        "num_bonds": num_bonds,
    }


def _validity_fail(error: str = "Invalid valence") -> Dict:
    return {
        "tool_name": "ValidityTool",
        "molecule_id": "mol_001",
        "is_valid": False,
        "error_message": error,
        "smiles_canonical": None,
        "num_atoms": 0,
        "num_bonds": 0,
    }


def _sa_pass(score: float = 2.1) -> Dict:
    return {
        "tool_name": "SAScoreTool",
        "molecule_id": "mol_001",
        "sa_score": score,
        "sa_decision": "PASS",
        "synthesizability_category": "easy",
        "sa_description": f"SA score {score:.1f} — easy to synthesize.",
        "complexity_breakdown": {"ring_complexity": 0.0},
        "warning_flags": [],
        "error_message": None,
        "execution_time_ms": 12.5,
    }


def _sa_flag(score: float = 6.5) -> Dict:
    return {
        "tool_name": "SAScoreTool",
        "molecule_id": "mol_001",
        "sa_score": score,
        "sa_decision": "FLAG",
        "synthesizability_category": "moderate",
        "sa_description": f"SA score {score:.1f} — borderline.",
        "complexity_breakdown": {"ring_complexity": 1.2},
        "warning_flags": ["2 stereocenters", "fused ring system"],
        "error_message": None,
        "execution_time_ms": 14.0,
    }


def _sa_discard(score: float = 8.0) -> Dict:
    return {
        "tool_name": "SAScoreTool",
        "molecule_id": "mol_001",
        "sa_score": score,
        "sa_decision": "DISCARD",
        "synthesizability_category": "very_difficult",
        "sa_description": f"SA score {score:.1f} — very difficult.",
        "complexity_breakdown": {},
        "warning_flags": ["macrocycle", "10 stereocenters"],
        "error_message": None,
        "execution_time_ms": 18.0,
    }


def _sa_error() -> Dict:
    return {
        "tool_name": "SAScoreTool",
        "molecule_id": "mol_001",
        "sa_score": None,
        "sa_decision": "ERROR",
        "synthesizability_category": None,
        "sa_description": None,
        "complexity_breakdown": None,
        "warning_flags": [],
        "error_message": "RDKit computation failed",
        "execution_time_ms": 1.0,
    }


def _sim_pass() -> Dict:
    return {
        "tool_name": "SimilarityTool",
        "molecule_id": "mol_001",
        "query_smiles": "CCO",
        "similarity_decision": "PASS",
        "nearest_neighbor_tanimoto": 0.4,
        "nearest_neighbor_source": None,
        "nearest_neighbor_id": None,
        "nearest_neighbor_name": None,
        "nearest_neighbor_smiles": None,
        "chembl_hits": [],
        "pubchem_hits": [],
        "flag_threshold_used": 0.85,
        "fingerprint_method": "Morgan",
        "query_fingerprint_hex": "abcdef",
        "apis_queried": ["ChEMBL", "PubChem"],
        "chembl_available": True,
        "pubchem_available": True,
        "execution_time_ms": 3200.0,
        "error_reason": None,
    }


def _sim_flag(tanimoto: float = 0.91) -> Dict:
    return {
        "tool_name": "SimilarityTool",
        "molecule_id": "mol_001",
        "query_smiles": "CCO",
        "similarity_decision": "FLAG",
        "nearest_neighbor_tanimoto": tanimoto,
        "nearest_neighbor_source": "ChEMBL",
        "nearest_neighbor_id": "CHEMBL545",
        "nearest_neighbor_name": "ALCOHOL",
        "nearest_neighbor_smiles": "CCO",
        "chembl_hits": [{"tanimoto": tanimoto, "id": "CHEMBL545", "name": "ALCOHOL"}],
        "pubchem_hits": [{"tanimoto": 0.85, "id": "702", "name": "ethanol"}],
        "flag_threshold_used": 0.85,
        "fingerprint_method": "Morgan",
        "query_fingerprint_hex": "abcdef",
        "apis_queried": ["ChEMBL", "PubChem"],
        "chembl_available": True,
        "pubchem_available": True,
        "execution_time_ms": 4100.0,
        "error_reason": None,
    }


def _sim_error() -> Dict:
    return {
        "tool_name": "SimilarityTool",
        "molecule_id": "mol_001",
        "query_smiles": "CCO",
        "similarity_decision": "ERROR",
        "nearest_neighbor_tanimoto": 0.0,
        "nearest_neighbor_source": None,
        "nearest_neighbor_id": None,
        "nearest_neighbor_name": None,
        "nearest_neighbor_smiles": None,
        "chembl_hits": [],
        "pubchem_hits": [],
        "flag_threshold_used": 0.85,
        "fingerprint_method": "Morgan",
        "query_fingerprint_hex": "",
        "apis_queried": [],
        "chembl_available": False,
        "pubchem_available": False,
        "execution_time_ms": 50.0,
        "error_reason": "Connection refused",
    }


# ============================================================================
# 1. TestRationaleBuilderValidity
# ============================================================================

class TestRationaleBuilderValidity:

    def setup_method(self):
        self.builder = RationaleBuilder()

    def test_validity_pass_section_status(self):
        state = _make_state(tool_results={"ValidityTool": _validity_pass()})
        exp = self.builder.build(state)
        section = next(s for s in exp.sections if s.tool == "Validity")
        assert section.status == "PASS"

    def test_validity_pass_headline_contains_atoms(self):
        state = _make_state(tool_results={"ValidityTool": _validity_pass(num_atoms=9)})
        exp = self.builder.build(state)
        section = next(s for s in exp.sections if s.tool == "Validity")
        assert "9 atoms" in section.headline

    def test_validity_pass_details_contain_canonical_smiles(self):
        state = _make_state(tool_results={"ValidityTool": _validity_pass(smiles="CCO")})
        exp = self.builder.build(state)
        section = next(s for s in exp.sections if s.tool == "Validity")
        assert any("CCO" in d for d in section.details)

    def test_validity_fail_status_is_discard(self):
        state = _make_state(
            decision_type_name="DISCARD",
            tool_results={"ValidityTool": _validity_fail()},
            terminated=True,
        )
        exp = self.builder.build(state)
        section = next(s for s in exp.sections if s.tool == "Validity")
        assert section.status == "DISCARD"

    def test_validity_fail_shows_error_message(self):
        state = _make_state(
            decision_type_name="DISCARD",
            tool_results={"ValidityTool": _validity_fail(error="Bad valence on N")},
            terminated=True,
        )
        exp = self.builder.build(state)
        section = next(s for s in exp.sections if s.tool == "Validity")
        assert any("Bad valence on N" in d for d in section.details)

    def test_validity_fail_no_sa_or_similarity_sections(self):
        """SA and Similarity sections should not appear when validity fails."""
        state = _make_state(
            decision_type_name="DISCARD",
            tool_results={"ValidityTool": _validity_fail()},
            terminated=True,
        )
        exp = self.builder.build(state)
        tool_names = [s.tool for s in exp.sections]
        assert "SA Score" not in tool_names
        assert "Similarity" not in tool_names

    def test_missing_validity_result_returns_skipped(self):
        state = _make_state(tool_results={})
        exp = self.builder.build(state)
        section = next(s for s in exp.sections if s.tool == "Validity")
        assert section.status == "SKIPPED"


# ============================================================================
# 2. TestRationaleBuilderSAScore
# ============================================================================

class TestRationaleBuilderSAScore:

    def setup_method(self):
        self.builder = RationaleBuilder()

    def _state_with_sa(self, sa_dict):
        return _make_state(tool_results={
            "ValidityTool": _validity_pass(),
            "SAScoreTool": sa_dict,
        })

    def test_sa_pass_status(self):
        exp = self.builder.build(self._state_with_sa(_sa_pass()))
        section = next(s for s in exp.sections if s.tool == "SA Score")
        assert section.status == "PASS"

    def test_sa_flag_status(self):
        exp = self.builder.build(self._state_with_sa(_sa_flag()))
        section = next(s for s in exp.sections if s.tool == "SA Score")
        assert section.status == "FLAG"

    def test_sa_discard_status(self):
        exp = self.builder.build(self._state_with_sa(_sa_discard()))
        section = next(s for s in exp.sections if s.tool == "SA Score")
        assert section.status == "DISCARD"

    def test_sa_error_status(self):
        exp = self.builder.build(self._state_with_sa(_sa_error()))
        section = next(s for s in exp.sections if s.tool == "SA Score")
        assert section.status == "ERROR"

    def test_sa_score_in_headline(self):
        exp = self.builder.build(self._state_with_sa(_sa_pass(score=3.14)))
        section = next(s for s in exp.sections if s.tool == "SA Score")
        assert "3.14" in section.headline

    def test_warning_flags_in_details(self):
        exp = self.builder.build(self._state_with_sa(_sa_flag()))
        section = next(s for s in exp.sections if s.tool == "SA Score")
        detail_text = " ".join(section.details)
        assert "stereocenters" in detail_text.lower()

    def test_thresholds_mentioned_in_details(self):
        exp = self.builder.build(self._state_with_sa(_sa_pass()))
        section = next(s for s in exp.sections if s.tool == "SA Score")
        detail_text = " ".join(section.details)
        assert "6.0" in detail_text

    def test_skipped_when_validity_failed(self):
        state = _make_state(
            decision_type_name="DISCARD",
            tool_results={"ValidityTool": _validity_fail()},
        )
        exp = self.builder.build(state)
        assert all(s.tool != "SA Score" for s in exp.sections)


# ============================================================================
# 3. TestRationaleBuilderSimilarity
# ============================================================================

class TestRationaleBuilderSimilarity:

    def setup_method(self):
        self.builder = RationaleBuilder()

    def _state_with_sim(self, sim_dict, sa_dict=None):
        return _make_state(tool_results={
            "ValidityTool": _validity_pass(),
            "SAScoreTool": sa_dict or _sa_pass(),
            "SimilarityTool": sim_dict,
        })

    def test_sim_pass_status(self):
        exp = self.builder.build(self._state_with_sim(_sim_pass()))
        section = next(s for s in exp.sections if s.tool == "Similarity")
        assert section.status == "PASS"

    def test_sim_flag_status(self):
        exp = self.builder.build(self._state_with_sim(_sim_flag()))
        section = next(s for s in exp.sections if s.tool == "Similarity")
        assert section.status == "FLAG"

    def test_sim_error_status(self):
        exp = self.builder.build(self._state_with_sim(_sim_error()))
        section = next(s for s in exp.sections if s.tool == "Similarity")
        assert section.status == "ERROR"

    def test_sim_flag_headline_contains_tanimoto(self):
        exp = self.builder.build(self._state_with_sim(_sim_flag(tanimoto=0.91)))
        section = next(s for s in exp.sections if s.tool == "Similarity")
        assert "0.910" in section.headline

    def test_sim_flag_headline_contains_compound_name(self):
        exp = self.builder.build(self._state_with_sim(_sim_flag()))
        section = next(s for s in exp.sections if s.tool == "Similarity")
        assert "ALCOHOL" in section.headline

    def test_escalated_warning_in_details(self):
        """Tanimoto >= 0.95 should trigger escalation warning."""
        exp = self.builder.build(self._state_with_sim(_sim_flag(tanimoto=1.0)))
        section = next(s for s in exp.sections if s.tool == "Similarity")
        detail_text = " ".join(section.details)
        assert "ESCALATED" in detail_text or "escalated" in detail_text.lower()

    def test_chembl_hit_count_in_details(self):
        exp = self.builder.build(self._state_with_sim(_sim_flag()))
        section = next(s for s in exp.sections if s.tool == "Similarity")
        detail_text = " ".join(section.details)
        assert "ChEMBL" in detail_text

    def test_sim_error_shows_reason(self):
        exp = self.builder.build(self._state_with_sim(_sim_error()))
        section = next(s for s in exp.sections if s.tool == "Similarity")
        detail_text = " ".join(section.details)
        assert "Connection refused" in detail_text

    def test_skipped_when_validity_failed(self):
        state = _make_state(
            decision_type_name="DISCARD",
            tool_results={"ValidityTool": _validity_fail()},
        )
        exp = self.builder.build(state)
        assert all(s.tool != "Similarity" for s in exp.sections)


# ============================================================================
# 4. TestRationaleBuilderSummary
# ============================================================================

class TestRationaleBuilderSummary:

    def setup_method(self):
        self.builder = RationaleBuilder()

    def test_summary_pass_mentions_all_checks(self):
        state = _make_state(
            decision_type_name="PASS",
            tool_results={
                "ValidityTool": _validity_pass(),
                "SAScoreTool": _sa_pass(),
                "SimilarityTool": _sim_pass(),
            },
        )
        exp = self.builder.build(state)
        assert "passed all checks" in exp.summary.lower()

    def test_summary_discard_invalid_mentions_invalid(self):
        state = _make_state(
            decision_type_name="DISCARD",
            tool_results={"ValidityTool": _validity_fail()},
            terminated=True,
        )
        exp = self.builder.build(state)
        assert "invalid" in exp.summary.lower()

    def test_summary_flag_similarity_mentions_ip(self):
        state = _make_state(
            decision_type_name="FLAG",
            tool_results={
                "ValidityTool": _validity_pass(),
                "SAScoreTool": _sa_pass(),
                "SimilarityTool": _sim_flag(),
            },
            flags=[{"reason": "Tanimoto 0.91", "source": "SimilarityTool"}],
        )
        exp = self.builder.build(state)
        assert "ip" in exp.summary.lower() or "flag" in exp.summary.lower()

    def test_summary_contains_molecule_id(self):
        state = _make_state(
            molecule_id="my_special_mol",
            decision_type_name="PASS",
            tool_results={
                "ValidityTool": _validity_pass(),
                "SAScoreTool": _sa_pass(),
                "SimilarityTool": _sim_pass(),
            },
        )
        exp = self.builder.build(state)
        assert "my_special_mol" in exp.summary


# ============================================================================
# 5. TestFormatText
# ============================================================================

class TestFormatText:

    def setup_method(self):
        self.builder = RationaleBuilder()

    def _full_pass_state(self):
        return _make_state(
            molecule_id="mol_text_test",
            decision_type_name="PASS",
            tool_results={
                "ValidityTool": _validity_pass(),
                "SAScoreTool": _sa_pass(),
                "SimilarityTool": _sim_pass(),
            },
        )

    def test_format_text_returns_string(self):
        exp = self.builder.build(self._full_pass_state())
        result = format_text(exp)
        assert isinstance(result, str)

    def test_format_text_contains_molecule_id(self):
        exp = self.builder.build(self._full_pass_state())
        result = format_text(exp)
        assert "mol_text_test" in result

    def test_format_text_contains_decision(self):
        exp = self.builder.build(self._full_pass_state())
        result = format_text(exp)
        assert "PASS" in result

    def test_format_text_contains_all_tool_names(self):
        exp = self.builder.build(self._full_pass_state())
        result = format_text(exp)
        assert "Validity" in result
        assert "SA Score" in result
        assert "Similarity" in result

    def test_format_dict_is_dict(self):
        exp = self.builder.build(self._full_pass_state())
        result = format_dict(exp)
        assert isinstance(result, dict)
        assert result["decision"] == "PASS"

    def test_format_text_shows_flags(self):
        state = _make_state(
            decision_type_name="FLAG",
            tool_results={
                "ValidityTool": _validity_pass(),
                "SAScoreTool": _sa_pass(),
                "SimilarityTool": _sim_flag(),
            },
            flags=[{"reason": "Tanimoto 0.91 >= 0.85", "source": "SimilarityTool"}],
        )
        exp = self.builder.build(state)
        result = format_text(exp)
        assert "FLAGS RAISED" in result
        assert "SimilarityTool" in result


# ============================================================================
# 6. TestRunRecordBuilder
# ============================================================================

class TestRunRecordBuilder:

    def setup_method(self):
        self.rb = RationaleBuilder()
        self.rrb = RunRecordBuilder()

    def _build_record(self, state):
        explanation = self.rb.build(state)
        return self.rrb.build(state, explanation)

    def test_record_has_run_id(self):
        state = _make_state(tool_results={"ValidityTool": _validity_pass(), "SAScoreTool": _sa_pass(), "SimilarityTool": _sim_pass()})
        record = self._build_record(state)
        assert record.run_id
        assert len(record.run_id) == 36  # UUID4 canonical form

    def test_record_molecule_id(self):
        state = _make_state(molecule_id="test_mol_007", tool_results={"ValidityTool": _validity_pass(), "SAScoreTool": _sa_pass(), "SimilarityTool": _sim_pass()})
        record = self._build_record(state)
        assert record.molecule_id == "test_mol_007"

    def test_record_final_decision_pass(self):
        state = _make_state(decision_type_name="PASS", tool_results={"ValidityTool": _validity_pass(), "SAScoreTool": _sa_pass(), "SimilarityTool": _sim_pass()})
        record = self._build_record(state)
        assert record.final_decision == "PASS"

    def test_record_final_decision_flag(self):
        state = _make_state(
            decision_type_name="FLAG",
            tool_results={"ValidityTool": _validity_pass(), "SAScoreTool": _sa_pass(), "SimilarityTool": _sim_flag()},
            flags=[{"reason": "Tanimoto high", "source": "SimilarityTool"}],
        )
        record = self._build_record(state)
        assert record.final_decision == "FLAG"

    def test_record_validity_fields(self):
        state = _make_state(tool_results={"ValidityTool": _validity_pass(num_atoms=9, num_bonds=8), "SAScoreTool": _sa_pass(), "SimilarityTool": _sim_pass()})
        record = self._build_record(state)
        assert record.is_valid is True
        assert record.num_atoms == 9
        assert record.num_bonds == 8

    def test_record_sa_fields(self):
        state = _make_state(tool_results={"ValidityTool": _validity_pass(), "SAScoreTool": _sa_flag(score=6.5), "SimilarityTool": _sim_pass()})
        record = self._build_record(state)
        assert record.sa_score == 6.5
        assert record.sa_decision == "FLAG"
        assert record.synthesizability_category == "moderate"

    def test_record_similarity_fields(self):
        state = _make_state(
            decision_type_name="FLAG",
            tool_results={"ValidityTool": _validity_pass(), "SAScoreTool": _sa_pass(), "SimilarityTool": _sim_flag(tanimoto=0.91)},
        )
        record = self._build_record(state)
        assert record.nn_tanimoto == 0.91
        assert record.nn_id == "CHEMBL545"
        assert record.chembl_hit_count == 1
        assert record.pubchem_hit_count == 1

    def test_record_escalated_flag(self):
        state = _make_state(
            decision_type_name="FLAG",
            tool_results={"ValidityTool": _validity_pass(), "SAScoreTool": _sa_pass(), "SimilarityTool": _sim_flag(tanimoto=1.0)},
        )
        record = self._build_record(state)
        assert record.similarity_escalated is True

    def test_record_not_escalated_below_threshold(self):
        state = _make_state(
            decision_type_name="FLAG",
            tool_results={"ValidityTool": _validity_pass(), "SAScoreTool": _sa_pass(), "SimilarityTool": _sim_flag(tanimoto=0.88)},
        )
        record = self._build_record(state)
        assert record.similarity_escalated is False

    def test_record_tools_executed_list(self):
        state = _make_state(tool_results={"ValidityTool": _validity_pass(), "SAScoreTool": _sa_pass(), "SimilarityTool": _sim_pass()})
        record = self._build_record(state)
        assert "ValidityTool" in record.tools_executed
        assert "SAScoreTool" in record.tools_executed
        assert "SimilarityTool" in record.tools_executed

    def test_record_ai_review_none_by_default(self):
        state = _make_state(tool_results={"ValidityTool": _validity_pass(), "SAScoreTool": _sa_pass(), "SimilarityTool": _sim_pass()})
        record = self._build_record(state)
        assert record.ai_review is None

    def test_record_ai_review_when_provided(self):
        ai = AIReviewRecord(
            model="claude-sonnet-4-20250514",
            ai_decision="FLAG",
            agrees=True,
            reasoning="Tanimoto of 0.91 clearly warrants IP review.",
            nuance=None,
            confidence="high",
            raw_response='{"ai_decision": "FLAG"}',
        )
        state = _make_state(tool_results={"ValidityTool": _validity_pass(), "SAScoreTool": _sa_pass(), "SimilarityTool": _sim_flag()})
        explanation = self.rb.build(state)
        record = self.rrb.build(state, explanation, ai_review=ai)
        assert record.ai_review is not None
        assert record.ai_review.model == "claude-sonnet-4-20250514"
        assert record.ai_review.agrees is True


# ============================================================================
# 7. TestRunRecordPersistence
# ============================================================================

class TestRunRecordPersistence:

    def _make_record(self, molecule_id: str = "mol_001", decision: str = "PASS") -> RunRecord:
        rb = RationaleBuilder()
        rrb = RunRecordBuilder()
        state = _make_state(
            molecule_id=molecule_id,
            decision_type_name=decision,
            tool_results={
                "ValidityTool": _validity_pass(),
                "SAScoreTool": _sa_pass(),
                "SimilarityTool": _sim_pass(),
            },
        )
        explanation = rb.build(state)
        return rrb.build(state, explanation)

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.jsonl"
            record = self._make_record()
            save(record, path)
            assert path.exists()

    def test_save_appends_multiple_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.jsonl"
            save(self._make_record("mol_001"), path)
            save(self._make_record("mol_002"), path)
            save(self._make_record("mol_003"), path)
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 3

    def test_each_line_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.jsonl"
            save(self._make_record("mol_001"), path)
            save(self._make_record("mol_002"), path)
            for line in path.read_text().strip().split("\n"):
                obj = json.loads(line)
                assert isinstance(obj, dict)

    def test_load_all_returns_run_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.jsonl"
            save(self._make_record("mol_001"), path)
            save(self._make_record("mol_002"), path)
            records = load_all(path)
            assert len(records) == 2
            assert all(isinstance(r, RunRecord) for r in records)

    def test_load_all_molecule_ids_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.jsonl"
            save(self._make_record("alpha"), path)
            save(self._make_record("beta"), path)
            records = load_all(path)
            ids = [r.molecule_id for r in records]
            assert "alpha" in ids
            assert "beta" in ids

    def test_load_all_nonexistent_file_returns_empty(self):
        records = load_all(Path("/tmp/does_not_exist_triflag.jsonl"))
        assert records == []

    def test_load_as_dicts_returns_dicts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.jsonl"
            save(self._make_record(), path)
            rows = load_as_dicts(path)
            assert len(rows) == 1
            assert isinstance(rows[0], dict)
            assert "molecule_id" in rows[0]

    def test_save_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "deep" / "nested" / "runs.jsonl"
            save(self._make_record(), path)
            assert path.exists()

    def test_load_all_skips_malformed_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.jsonl"
            save(self._make_record("good"), path)
            with path.open("a") as fh:
                fh.write("{this is not json}\n")
            save(self._make_record("also_good"), path)
            records = load_all(path)
            assert len(records) == 2
            assert all(r.molecule_id in ("good", "also_good") for r in records)


# ============================================================================
# 8. TestRunRecordRoundTrip
# ============================================================================

class TestRunRecordRoundTrip:

    def _make_record(self) -> RunRecord:
        rb = RationaleBuilder()
        rrb = RunRecordBuilder()
        state = _make_state(
            molecule_id="roundtrip_mol",
            decision_type_name="FLAG",
            tool_results={
                "ValidityTool": _validity_pass(),
                "SAScoreTool": _sa_flag(),
                "SimilarityTool": _sim_flag(),
            },
            flags=[{"reason": "SA borderline", "source": "SAScoreTool"}],
        )
        explanation = rb.build(state)
        ai = AIReviewRecord(
            model="claude-sonnet-4-20250514",
            ai_decision="FLAG",
            agrees=True,
            reasoning="Borderline SA and high similarity both warrant review.",
            nuance="SA score may benefit from analogue comparison.",
            confidence="high",
            raw_response='{}',
        )
        return rrb.build(state, explanation, ai_review=ai)

    def test_to_dict_is_json_serialisable(self):
        record = self._make_record()
        d = record.to_dict()
        # Should not raise
        serialised = json.dumps(d)
        assert isinstance(serialised, str)

    def test_from_dict_restores_molecule_id(self):
        record = self._make_record()
        d = record.to_dict()
        restored = RunRecord.from_dict(d)
        assert restored.molecule_id == record.molecule_id

    def test_from_dict_restores_decision(self):
        record = self._make_record()
        d = record.to_dict()
        restored = RunRecord.from_dict(d)
        assert restored.final_decision == record.final_decision

    def test_from_dict_restores_ai_review(self):
        record = self._make_record()
        d = record.to_dict()
        restored = RunRecord.from_dict(d)
        assert restored.ai_review is not None
        assert restored.ai_review.model == "claude-sonnet-4-20250514"
        assert restored.ai_review.agrees is True

    def test_from_dict_restores_sa_score(self):
        record = self._make_record()
        d = record.to_dict()
        restored = RunRecord.from_dict(d)
        assert restored.sa_score == record.sa_score


# ============================================================================
# 9. TestEarlyTermination
# ============================================================================

class TestEarlyTermination:

    def setup_method(self):
        self.builder = RationaleBuilder()

    def test_early_termination_flag_set(self):
        state = _make_state(
            decision_type_name="DISCARD",
            tool_results={"ValidityTool": _validity_fail()},
            terminated=True,
            termination_reason="Invalid SMILES — pipeline halted.",
        )
        exp = self.builder.build(state)
        assert exp.early_termination is True

    def test_early_termination_reason_preserved(self):
        state = _make_state(
            decision_type_name="DISCARD",
            tool_results={"ValidityTool": _validity_fail()},
            terminated=True,
            termination_reason="Invalid SMILES — pipeline halted.",
        )
        exp = self.builder.build(state)
        assert exp.termination_reason == "Invalid SMILES — pipeline halted."

    def test_format_text_shows_early_exit(self):
        state = _make_state(
            decision_type_name="DISCARD",
            tool_results={"ValidityTool": _validity_fail()},
            terminated=True,
            termination_reason="Invalid SMILES.",
        )
        exp = self.builder.build(state)
        text = format_text(exp)
        assert "Early exit" in text or "early" in text.lower()

    def test_sa_skipped_on_early_termination(self):
        state = _make_state(
            decision_type_name="DISCARD",
            tool_results={"ValidityTool": _validity_fail()},
            terminated=True,
        )
        exp = self.builder.build(state)
        tool_names = [s.tool for s in exp.sections]
        assert "SA Score" not in tool_names

    def test_run_record_captures_early_termination(self):
        rb = RationaleBuilder()
        rrb = RunRecordBuilder()
        state = _make_state(
            decision_type_name="DISCARD",
            tool_results={"ValidityTool": _validity_fail()},
            terminated=True,
            termination_reason="Invalid SMILES.",
        )
        explanation = rb.build(state)
        record = rrb.build(state, explanation)
        assert record.early_termination is True
        assert record.termination_reason == "Invalid SMILES."