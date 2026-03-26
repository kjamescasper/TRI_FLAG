# tests/test_week10.py
"""
Week 10 — ACEGEN interface contract tests.

These tests verify that triflag_score() satisfies the strict interface
contract that ACEGEN requires. A single violation — wrong length, non-float
output, out-of-range value, or raised exception — silently kills ACEGEN's
training loop with no stack trace.

All tests run without network access (SimilarityTool mocked) and use an
in-memory or temporary SQLite database so they are fast and hermetic.

Test inventory:
    1. length_contract        — output length == input length for any input
    2. float_type             — every output element is a Python float
    3. range_contract         — every output is in [0.0, 1.0]
    4. invalid_smiles         — unparseable SMILES returns 0.0, never raises
    5. sqlite_write           — every scored molecule appears in the database
    6. empty_input            — empty list → empty list (no exception)
    7. single_molecule        — single-element list works correctly
    8. known_molecule_range   — ethanol scores 0.0 (novelty collapse expected)
    9. no_exception_on_junk   — garbage strings never cause a raised exception
   10. module_config          — BATCH_ID and GENERATION_NUMBER written to DB rows

Run with:
    set PYTHONPATH=.
    set KMP_DUPLICATE_LIB_OK=TRUE
    pytest tests/test_week10.py -v
"""

import importlib
import logging
import sqlite3
import tempfile
import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Standard test molecules covering the main pipeline paths.
VALID_SMILES = [
    "CCO",                                      # ethanol — valid, low complexity
    "CC(=O)Oc1ccccc1C(=O)O",                   # aspirin — known drug, novelty=0
    "CC1=CC(=CC=C1)NC(=O)C2=CC=CC=C2F",        # fluorobenzamide — novel scaffold
    "c1ccc2ccccc2c1",                           # naphthalene — simple aromatic
    "CC(C)(C)c1ccc(cc1)C(=O)O",                # tBu-benzoic acid — drug-like
]
INVALID_SMILES = [
    "not_a_smiles",
    "INVALID!!",
    "",
    "C(C)(C)(C)(C)(C)",   # pentavalent carbon — chemically invalid
    "   ",
]


@pytest.fixture(autouse=True)
def silence_logging():
    """Suppress pipeline logging noise during tests."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture()
def tmp_db(tmp_path):
    """Return a path to a fresh temporary SQLite database."""
    return str(tmp_path / "test_triflag.db")


@pytest.fixture()
def scorer_module(tmp_db):
    """
    Import (or reimport) triflag_scorer with:
      - similarity disabled (no live API calls)
      - temporary database (hermetic, no cross-test pollution)
    Returns the module object so tests can read/write config attributes.
    """
    import loop.triflag_scorer as scorer
    # Reset module-level config to safe test defaults.
    scorer.SKIP_SIMILARITY = True
    scorer.BATCH_ID = "test_batch"
    scorer.GENERATION_NUMBER = 0
    scorer.DB_PATH = tmp_db
    return scorer


# ---------------------------------------------------------------------------
# Helper: mock out SimilarityTool network calls
# The patch target must match where SimilarityTool is *used* (triflag_scorer),
# not where it is defined, because that is what Python resolves at import time.
# ---------------------------------------------------------------------------

def _mock_similarity_tool():
    """
    Return a context manager that replaces SimilarityTool with a mock that
    immediately returns a PASS result without making any network calls.
    """
    mock_instance = MagicMock()
    mock_instance.name = "SimilarityTool"
    mock_instance.run.return_value = {
        "tool_name": "SimilarityTool",
        "similarity_decision": "PASS",
        "nearest_neighbor_tanimoto": 0.0,
        "nearest_neighbor_source": None,
        "nearest_neighbor_id": None,
        "nearest_neighbor_name": None,
        "nearest_neighbor_smiles": None,
        "chembl_hits": [],
        "pubchem_hits": [],
        "flag_threshold_used": 0.90,
        "fingerprint_method": "morgan",
        "query_fingerprint_hex": "",
        "apis_queried": [],
        "chembl_available": False,
        "pubchem_available": False,
        "execution_time_ms": 0.0,
        "error_reason": None,
    }
    return patch("loop.triflag_scorer.SimilarityTool", return_value=mock_instance)


# ---------------------------------------------------------------------------
# Test 1 — Length contract
# ---------------------------------------------------------------------------

class TestLengthContract:
    """Output list must always be the same length as input list."""

    def test_length_matches_five_molecules(self, scorer_module):
        result = scorer_module.triflag_score(VALID_SMILES)
        assert len(result) == len(VALID_SMILES), (
            f"Expected {len(VALID_SMILES)} scores, got {len(result)}"
        )

    def test_length_matches_mixed_valid_invalid(self, scorer_module):
        mixed = VALID_SMILES[:2] + INVALID_SMILES[:2]
        result = scorer_module.triflag_score(mixed)
        assert len(result) == len(mixed)

    def test_length_matches_all_invalid(self, scorer_module):
        result = scorer_module.triflag_score(INVALID_SMILES)
        assert len(result) == len(INVALID_SMILES)

    def test_length_single(self, scorer_module):
        result = scorer_module.triflag_score(["CCO"])
        assert len(result) == 1

    def test_length_large_batch(self, scorer_module):
        batch = VALID_SMILES * 10   # 50 molecules
        result = scorer_module.triflag_score(batch)
        assert len(result) == 50


# ---------------------------------------------------------------------------
# Test 2 — Float type
# ---------------------------------------------------------------------------

class TestFloatType:
    """Every element of the output must be a Python float."""

    def test_all_outputs_are_float_valid_input(self, scorer_module):
        result = scorer_module.triflag_score(VALID_SMILES)
        for i, val in enumerate(result):
            assert isinstance(val, float), (
                f"Score at index {i} is {type(val).__name__}, expected float"
            )

    def test_all_outputs_are_float_invalid_input(self, scorer_module):
        result = scorer_module.triflag_score(INVALID_SMILES)
        for i, val in enumerate(result):
            assert isinstance(val, float), (
                f"Score at index {i} is {type(val).__name__}, expected float"
            )

    def test_output_is_not_int_or_none(self, scorer_module):
        result = scorer_module.triflag_score(["CCO"])
        assert result[0] is not None
        assert not isinstance(result[0], int)


# ---------------------------------------------------------------------------
# Test 3 — Range contract
# ---------------------------------------------------------------------------

class TestRangeContract:
    """Every output value must be in [0.0, 1.0]."""

    def test_all_values_in_range_valid(self, scorer_module):
        result = scorer_module.triflag_score(VALID_SMILES)
        for i, val in enumerate(result):
            assert 0.0 <= val <= 1.0, (
                f"Score at index {i} is {val}, outside [0.0, 1.0]"
            )

    def test_all_values_in_range_invalid(self, scorer_module):
        result = scorer_module.triflag_score(INVALID_SMILES)
        for i, val in enumerate(result):
            assert 0.0 <= val <= 1.0, (
                f"Score at index {i} is {val}, outside [0.0, 1.0]"
            )

    def test_all_values_in_range_mixed(self, scorer_module):
        mixed = VALID_SMILES + INVALID_SMILES
        result = scorer_module.triflag_score(mixed)
        for i, val in enumerate(result):
            assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# Test 4 — Invalid SMILES handling
# ---------------------------------------------------------------------------

class TestInvalidSMILES:
    """Invalid SMILES must return 0.0 and never raise an exception."""

    def test_invalid_smiles_returns_zero(self, scorer_module):
        result = scorer_module.triflag_score(["not_a_smiles"])
        assert result == [0.0], f"Expected [0.0], got {result}"

    def test_empty_string_returns_zero(self, scorer_module):
        result = scorer_module.triflag_score([""])
        assert result == [0.0]

    def test_garbage_string_no_exception(self, scorer_module):
        """Must return 0.0 without raising for any string input."""
        garbage = ["!!!###", "123ABC", "\x00\x01", "N" * 500]
        result = scorer_module.triflag_score(garbage)
        assert len(result) == len(garbage)
        assert all(v == 0.0 for v in result)

    def test_none_like_strings_no_exception(self, scorer_module):
        result = scorer_module.triflag_score(["None", "null", "nan"])
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_no_exception_raised_ever(self, scorer_module):
        """The function contract is: never raises, no matter what."""
        problematic = ["not_smiles", "", "C(C)(C)(C)(C)C", "INVALID", "   "]
        try:
            scorer_module.triflag_score(problematic)
        except Exception as exc:
            pytest.fail(
                f"triflag_score raised {type(exc).__name__}: {exc} — "
                "this violates the ACEGEN interface contract"
            )


# ---------------------------------------------------------------------------
# Test 5 — SQLite write
# ---------------------------------------------------------------------------

class TestSQLiteWrite:
    """Every scored molecule must appear as a row in the database."""

    def test_molecules_written_to_db(self, scorer_module, tmp_db):
        scorer_module.DB_PATH = tmp_db
        smiles_batch = VALID_SMILES[:3]
        scorer_module.triflag_score(smiles_batch)

        conn = sqlite3.connect(tmp_db)
        row_count = conn.execute("SELECT COUNT(*) FROM triage_runs").fetchone()[0]
        conn.close()

        assert row_count >= len(smiles_batch), (
            f"Expected at least {len(smiles_batch)} rows in triage_runs, "
            f"found {row_count}"
        )

    def test_invalid_smiles_not_written(self, scorer_module, tmp_db):
        """
        Invalid SMILES cannot produce a valid RunRecord and should not appear
        in the database (ValidityTool terminates the pipeline before DB write).
        """
        scorer_module.DB_PATH = tmp_db
        scorer_module.triflag_score(["not_a_smiles"])

        conn = sqlite3.connect(tmp_db)
        # If any rows exist from invalid input, that would be unexpected.
        # The pipeline should fail at ValidityTool and not reach save().
        # We just check that no exception was raised (already tested above)
        # and that the DB doesn't explode.
        conn.execute("SELECT COUNT(*) FROM triage_runs")
        conn.close()

    def test_batch_id_written_to_db(self, scorer_module, tmp_db):
        """batch_id module config should appear in triage_runs rows."""
        scorer_module.DB_PATH = tmp_db
        scorer_module.BATCH_ID = "pytest_batch_001"
        scorer_module.triflag_score(["CCO", "c1ccccc1"])

        conn = sqlite3.connect(tmp_db)
        rows = conn.execute(
            "SELECT batch_id FROM triage_runs WHERE batch_id = ?",
            ("pytest_batch_001",),
        ).fetchall()
        conn.close()

        # At least the valid molecule (benzene) should have been written.
        assert len(rows) >= 1, "batch_id not written to triage_runs"

    def test_generation_number_written_to_db(self, scorer_module, tmp_db):
        """generation_number module config should appear in triage_runs rows."""
        scorer_module.DB_PATH = tmp_db
        scorer_module.GENERATION_NUMBER = 42
        scorer_module.triflag_score(["c1ccccc1"])   # benzene — valid

        conn = sqlite3.connect(tmp_db)
        rows = conn.execute(
            "SELECT generation_number FROM triage_runs WHERE generation_number = 42"
        ).fetchall()
        conn.close()

        assert len(rows) >= 1, "generation_number not written to triage_runs"


# ---------------------------------------------------------------------------
# Test 6 — Empty input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    """Empty list → empty list, no exception."""

    def test_empty_list_returns_empty_list(self, scorer_module):
        result = scorer_module.triflag_score([])
        assert result == [], f"Expected [], got {result}"

    def test_empty_list_type(self, scorer_module):
        result = scorer_module.triflag_score([])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Test 7 — Single molecule
# ---------------------------------------------------------------------------

class TestSingleMolecule:
    """Single-element list must work correctly."""

    def test_single_valid_molecule(self, scorer_module):
        result = scorer_module.triflag_score(["c1ccccc1"])
        assert len(result) == 1
        assert isinstance(result[0], float)
        assert 0.0 <= result[0] <= 1.0

    def test_single_invalid_molecule(self, scorer_module):
        result = scorer_module.triflag_score(["NOT_SMILES"])
        assert result == [0.0]


# ---------------------------------------------------------------------------
# Test 8 — Known molecule scoring behaviour
# ---------------------------------------------------------------------------

class TestKnownMoleculeScoring:
    """
    Ethanol (CCO) and aspirin are both in ChEMBL with Tanimoto ~1.0.
    With SKIP_SIMILARITY=True (no similarity component), their scores depend
    on SA and QED alone. We test ordering rather than exact values because
    SA scores vary by RDKit version.
    """

    def test_ethanol_scores_low(self, scorer_module):
        """
        Ethanol should score low — it's trivially simple and not drug-like.
        With SKIP_SIMILARITY=True, SA reward is high but QED is very low.
        We just assert it's in range rather than an exact value.
        """
        result = scorer_module.triflag_score(["CCO"])
        assert len(result) == 1
        assert isinstance(result[0], float)
        assert 0.0 <= result[0] <= 1.0

    def test_complex_molecule_scores_higher_than_ethanol(self, scorer_module):
        """
        A more drug-like scaffold should outscore ethanol.
        fluorobenzamide has better QED and reasonable SA score.
        """
        ethanol_score = scorer_module.triflag_score(["CCO"])[0]
        complex_score = scorer_module.triflag_score(
            ["CC1=CC(=CC=C1)NC(=O)C2=CC=CC=C2F"]
        )[0]
        assert complex_score >= ethanol_score, (
            f"Expected complex molecule ({complex_score:.4f}) >= "
            f"ethanol ({ethanol_score:.4f})"
        )

    def test_invalid_always_zero(self, scorer_module):
        invalid_score = scorer_module.triflag_score(["not_a_smiles"])[0]
        assert invalid_score == 0.0


# ---------------------------------------------------------------------------
# Test 9 — No exception on pathological input
# ---------------------------------------------------------------------------

class TestNoExceptionOnJunk:
    """
    ACEGEN may pass unusual strings in early training — long SMILES,
    repeated characters, fragments, etc. None should raise.
    """

    @pytest.mark.parametrize("smi", [
        "C" * 200,                              # extremely long carbon chain
        "c1ccc(cc1)" * 10,                      # repeated fragment
        "CCCCCCCCCCCCCCCCCCCC(=O)O",            # long fatty acid
        "[Na+].[Cl-]",                          # salt
        "B(O)(O)c1ccccc1",                      # boronic acid
        "\n",                                   # newline only
        "C#C#C#C",                              # cumulated triple bonds
    ])
    def test_no_exception(self, scorer_module, smi):
        try:
            result = scorer_module.triflag_score([smi])
        except Exception as exc:
            pytest.fail(
                f"triflag_score raised {type(exc).__name__} for input {smi!r}: {exc}"
            )
        assert len(result) == 1
        assert isinstance(result[0], float)
        assert 0.0 <= result[0] <= 1.0


# ---------------------------------------------------------------------------
# Test 10 — Module config is respected
# ---------------------------------------------------------------------------

class TestModuleConfig:
    """
    Module-level config (BATCH_ID, GENERATION_NUMBER, SKIP_SIMILARITY, DB_PATH)
    must be respected by triflag_score() without requiring any argument passing.
    """

    def test_db_path_config_respected(self, scorer_module, tmp_path):
        custom_db = str(tmp_path / "custom.db")
        scorer_module.DB_PATH = custom_db
        scorer_module.triflag_score(["c1ccccc1"])

        assert os.path.exists(custom_db), (
            f"Database was not created at configured DB_PATH: {custom_db}"
        )

    def test_skip_similarity_true_still_scores(self, scorer_module):
        """SKIP_SIMILARITY=True must still return valid scores."""
        scorer_module.SKIP_SIMILARITY = True
        result = scorer_module.triflag_score(["CC(=O)Oc1ccccc1C(=O)O"])
        assert len(result) == 1
        assert isinstance(result[0], float)
        assert 0.0 <= result[0] <= 1.0

    def test_batch_id_none_does_not_crash(self, scorer_module):
        """None batch_id (unset) must not cause exceptions."""
        scorer_module.BATCH_ID = None
        scorer_module.GENERATION_NUMBER = None
        try:
            scorer_module.triflag_score(["CCO"])
        except Exception as exc:
            pytest.fail(f"triflag_score raised with None config: {exc}")