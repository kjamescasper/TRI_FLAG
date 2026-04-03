# tests/test_week11.py
"""
Week 11 test suite — DeepPurpose S_act integration.

All tests mock DeepPurpose to avoid network calls and model downloads.
No test in this file touches the real DTI model.

Architecture notes (important for mock strategy)
-------------------------------------------------
* deeppurpose_model.py uses LAZY imports — `from DeepPurpose import DTI`
  only runs inside load_model() / predict_binding(), never at module level.
  There is NO module-level `_DTI` attribute to patch.  Instead we inject
  a mock DTI via sys.modules so the `from DeepPurpose import DTI` line
  inside load_model() picks it up.

* scoring.py gates DeepPurpose behind TRIFLAG_ENABLE_DEEPPURPOSE=1.
  In normal pytest runs the env var is "0", so _DEEPPURPOSE_AVAILABLE=False
  and predict_binding/normalise_affinity remain None.  Tests that need
  S_act to be active must patch _DEEPPURPOSE_AVAILABLE=True AND supply
  callable replacements for predict_binding / normalise_affinity.

Coverage
--------
* normalise_affinity() math: MIN→0.0, MAX→1.0, midpoint→0.5
* normalise_affinity() clamping: above MAX→1.0, below MIN→0.0
* predict_binding() with mocked model returns float
* predict_binding() with invalid SMILES returns AFFINITY_MIN (no exception)
* Model load failure → graceful fallback, WARNING logged
* compute_reward() with S_act=0.0 collapses full reward to 0.0
* compute_reward() with S_act=1.0 matches three-component formula
* DeepPurpose unavailable → S_act defaults to 1.0
* target_config constants correct
"""

from __future__ import annotations

import logging
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: inject a fake DeepPurpose.DTI into sys.modules so lazy imports work
# ---------------------------------------------------------------------------

def _make_fake_dti_module(score: float = 7.5) -> MagicMock:
    """
    Return a MagicMock that behaves like the DeepPurpose.DTI module.
    virtual_screening returns a one-row DataFrame with the given score.
    model_pretrained returns a MagicMock model object.
    """
    import pandas as pd
    fake_dti = MagicMock()
    fake_dti.virtual_screening.return_value = pd.DataFrame({"Score": [score]})
    fake_dti.model_pretrained.return_value = MagicMock()
    return fake_dti


def _inject_dti(fake_dti: MagicMock):
    """
    Context manager: temporarily replace DeepPurpose.DTI in sys.modules.
    Also ensures DeepPurpose package stub exists so `from DeepPurpose import DTI`
    resolves correctly inside load_model().
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        # Build a minimal fake DeepPurpose package if needed
        fake_pkg = sys.modules.get("DeepPurpose") or MagicMock()
        fake_pkg.DTI = fake_dti
        old_pkg = sys.modules.get("DeepPurpose")
        old_dti = sys.modules.get("DeepPurpose.DTI")
        sys.modules["DeepPurpose"] = fake_pkg
        sys.modules["DeepPurpose.DTI"] = fake_dti
        try:
            yield fake_dti
        finally:
            if old_pkg is None:
                sys.modules.pop("DeepPurpose", None)
            else:
                sys.modules["DeepPurpose"] = old_pkg
            if old_dti is None:
                sys.modules.pop("DeepPurpose.DTI", None)
            else:
                sys.modules["DeepPurpose.DTI"] = old_dti

    return _ctx()


# ---------------------------------------------------------------------------
# 1. normalise_affinity — pure math, no mocking needed
# ---------------------------------------------------------------------------

class TestNormaliseAffinity:
    """Pure unit tests for the normalisation formula."""

    def setup_method(self):
        from target.deeppurpose_model import normalise_affinity
        from target.target_config import AFFINITY_MAX, AFFINITY_MIN
        self.fn = normalise_affinity
        self.MIN = AFFINITY_MIN
        self.MAX = AFFINITY_MAX

    def test_at_min_returns_zero(self):
        assert self.fn(self.MIN) == pytest.approx(0.0)

    def test_at_max_returns_one(self):
        assert self.fn(self.MAX) == pytest.approx(1.0)

    def test_at_midpoint_returns_half(self):
        midpoint = (self.MIN + self.MAX) / 2.0  # 7.0
        assert self.fn(midpoint) == pytest.approx(0.5)

    def test_above_max_clamps_to_one(self):
        assert self.fn(self.MAX + 5.0) == pytest.approx(1.0)

    def test_well_above_max_clamps_to_one(self):
        assert self.fn(100.0) == pytest.approx(1.0)

    def test_below_min_clamps_to_zero(self):
        assert self.fn(self.MIN - 1.0) == pytest.approx(0.0)

    def test_well_below_min_clamps_to_zero(self):
        assert self.fn(-99.0) == pytest.approx(0.0)

    def test_quarter_point(self):
        assert self.fn(self.MIN + 1.5) == pytest.approx(0.25)

    def test_three_quarter_point(self):
        assert self.fn(self.MIN + 4.5) == pytest.approx(0.75)

    def test_returns_float(self):
        assert isinstance(self.fn(7.0), float)


# ---------------------------------------------------------------------------
# 2. predict_binding — lazy-import mocking via sys.modules injection
# ---------------------------------------------------------------------------

class TestPredictBinding:
    """Tests for predict_binding() with DeepPurpose mocked via sys.modules."""

    def _reset_singleton(self, mod):
        """Reset module singleton state so load_model() tries again."""
        mod._MODEL_SINGLETON = None
        mod._MODEL_LOAD_FAILED = False
        mod._MODEL_AVAILABLE = False

    def test_valid_smiles_returns_float_in_range(self):
        """Mock returning pIC50 7.5 — predict_binding should return 7.5."""
        from target import deeppurpose_model as mod
        self._reset_singleton(mod)

        fake_dti = _make_fake_dti_module(score=7.5)
        with _inject_dti(fake_dti):
            result = mod.predict_binding("CCO")

        assert isinstance(result, float)
        assert result == pytest.approx(7.5)

        # Restore
        self._reset_singleton(mod)

    def test_invalid_smiles_returns_affinity_min(self):
        """If virtual_screening raises, predict_binding returns AFFINITY_MIN."""
        from target import deeppurpose_model as mod
        from target.target_config import AFFINITY_MIN
        self._reset_singleton(mod)

        fake_dti = _make_fake_dti_module()
        fake_dti.virtual_screening.side_effect = Exception("bad SMILES")
        with _inject_dti(fake_dti):
            result = mod.predict_binding("NOT_A_SMILES_$$$$")

        assert result == pytest.approx(AFFINITY_MIN)
        self._reset_singleton(mod)

    def test_empty_string_returns_affinity_min(self):
        from target.deeppurpose_model import predict_binding
        from target.target_config import AFFINITY_MIN
        assert predict_binding("") == pytest.approx(AFFINITY_MIN)

    def test_none_input_returns_affinity_min(self):
        from target.deeppurpose_model import predict_binding
        from target.target_config import AFFINITY_MIN
        assert predict_binding(None) == pytest.approx(AFFINITY_MIN)  # type: ignore[arg-type]

    def test_model_none_returns_affinity_min(self):
        """When load_model() returns None (no DTI in sys.modules), return AFFINITY_MIN."""
        from target import deeppurpose_model as mod
        from target.target_config import AFFINITY_MIN
        self._reset_singleton(mod)

        # Don't inject any DTI — load_model will fail gracefully
        result = mod.predict_binding("CCO")
        assert result == pytest.approx(AFFINITY_MIN)
        self._reset_singleton(mod)

    def test_virtual_screening_exception_returns_affinity_min(self):
        """RuntimeError in virtual_screening is swallowed, returns AFFINITY_MIN."""
        from target import deeppurpose_model as mod
        from target.target_config import AFFINITY_MIN
        self._reset_singleton(mod)

        fake_dti = _make_fake_dti_module()
        fake_dti.virtual_screening.side_effect = RuntimeError("CUDA OOM")
        with _inject_dti(fake_dti):
            result = mod.predict_binding("c1ccccc1")

        assert result == pytest.approx(AFFINITY_MIN)
        self._reset_singleton(mod)


# ---------------------------------------------------------------------------
# 3. load_model — failure path
# ---------------------------------------------------------------------------

class TestLoadModel:
    """Tests for load_model() graceful degradation."""

    def _reset_singleton(self, mod):
        mod._MODEL_SINGLETON = None
        mod._MODEL_LOAD_FAILED = False
        mod._MODEL_AVAILABLE = False

    def test_load_failure_returns_none_and_warns(self, caplog):
        """If model_pretrained raises, load_model returns None and logs WARNING."""
        from target import deeppurpose_model as mod
        self._reset_singleton(mod)

        fake_dti = _make_fake_dti_module()
        fake_dti.model_pretrained.side_effect = RuntimeError("download failed")

        with _inject_dti(fake_dti), \
             caplog.at_level(logging.WARNING, logger="target.deeppurpose_model"):
            result = mod.load_model()

        self._reset_singleton(mod)

        assert result is None
        assert any("load failed" in r.message.lower() for r in caplog.records)

    def test_load_failure_sets_load_failed_flag(self):
        """After a failed load, _MODEL_LOAD_FAILED=True prevents retry."""
        from target import deeppurpose_model as mod
        self._reset_singleton(mod)

        fake_dti = _make_fake_dti_module()
        fake_dti.model_pretrained.side_effect = RuntimeError("network error")

        with _inject_dti(fake_dti):
            mod.load_model()
            assert mod._MODEL_LOAD_FAILED is True

        self._reset_singleton(mod)

    def test_deeppurpose_unavailable_returns_none(self):
        """When DeepPurpose is not in sys.modules, load_model returns None gracefully."""
        from target import deeppurpose_model as mod
        self._reset_singleton(mod)

        # Ensure DeepPurpose is absent from sys.modules
        saved = sys.modules.pop("DeepPurpose", None)
        saved_dti = sys.modules.pop("DeepPurpose.DTI", None)
        try:
            result = mod.load_model()
        finally:
            if saved is not None:
                sys.modules["DeepPurpose"] = saved
            if saved_dti is not None:
                sys.modules["DeepPurpose.DTI"] = saved_dti
            self._reset_singleton(mod)

        assert result is None


# ---------------------------------------------------------------------------
# 4. compute_reward integration with S_act
# ---------------------------------------------------------------------------

class TestComputeRewardWithSAct:
    """
    Tests for scoring.compute_reward() with S_act controlled via patching.

    Since scoring.py gates DeepPurpose behind _DEEPPURPOSE_AVAILABLE, tests
    that need S_act to fire must patch both:
      - reporting.scoring._DEEPPURPOSE_AVAILABLE = True
      - reporting.scoring.predict_binding = <callable returning desired value>
      - reporting.scoring.normalise_affinity = <callable returning desired S_act>
    """

    def _make_record(self, sa_score=2.5, nn_tanimoto=0.3, smiles="c1ccccc1C"):
        from reporting.run_record import RunRecord
        r = RunRecord.__new__(RunRecord)
        r.smiles_canonical = smiles
        r.sa_score = sa_score
        r.nn_tanimoto = nn_tanimoto
        r.is_valid = True
        r.final_decision = "PASS"
        r.qed = None
        r.s_sa = None
        r.s_nov = None
        r.s_qed = None
        r.s_act = None
        r.reward = None
        return r

    def test_s_act_zero_collapses_reward_to_zero(self):
        """S_act=0.0 must zero out the entire reward (multiplicative)."""
        from reporting import scoring
        record = self._make_record()

        with patch.object(scoring, "_DEEPPURPOSE_AVAILABLE", True), \
             patch.object(scoring, "predict_binding", return_value=4.0), \
             patch.object(scoring, "normalise_affinity", return_value=0.0):
            result = scoring.compute_reward(record)

        assert result.reward == pytest.approx(0.0)
        assert result.s_act == pytest.approx(0.0)

    def test_s_act_one_matches_three_component_formula(self):
        """S_act=1.0 must produce same reward as three-component formula."""
        from reporting import scoring
        record = self._make_record()

        with patch.object(scoring, "_DEEPPURPOSE_AVAILABLE", True), \
             patch.object(scoring, "predict_binding", return_value=10.0), \
             patch.object(scoring, "normalise_affinity", return_value=1.0):
            result_with_sact = scoring.compute_reward(record)

        with patch.object(scoring, "_DEEPPURPOSE_AVAILABLE", False):
            result_without_sact = scoring.compute_reward(record)

        assert result_with_sact.reward == pytest.approx(result_without_sact.reward, abs=1e-9)

    def test_s_act_midpoint_halves_reward(self):
        """S_act=0.5 should halve the three-component reward."""
        from reporting import scoring
        record = self._make_record()

        with patch.object(scoring, "_DEEPPURPOSE_AVAILABLE", False):
            base = scoring.compute_reward(record)

        with patch.object(scoring, "_DEEPPURPOSE_AVAILABLE", True), \
             patch.object(scoring, "predict_binding", return_value=7.0), \
             patch.object(scoring, "normalise_affinity", return_value=0.5):
            result = scoring.compute_reward(record)

        assert result.reward == pytest.approx(base.reward * 0.5, abs=1e-9)

    def test_deeppurpose_unavailable_defaults_s_act_to_one(self):
        """When DeepPurpose is absent, S_act must be 1.0 and reward > 0."""
        from reporting import scoring
        record = self._make_record()

        with patch.object(scoring, "_DEEPPURPOSE_AVAILABLE", False):
            result = scoring.compute_reward(record)

        assert result.s_act == pytest.approx(1.0)
        assert result.reward > 0.0

    def test_s_act_field_present_in_reward_result(self):
        """RewardResult must expose an s_act field."""
        from reporting.scoring import RewardResult
        r = RewardResult(reward=0.5, s_sa=0.8, s_nov=0.9, s_qed=0.7, s_act=1.0)
        assert hasattr(r, "s_act")
        assert r.s_act == pytest.approx(1.0)

    def test_compute_reward_returns_reward_result_type(self):
        from reporting import scoring
        from reporting.scoring import RewardResult
        record = self._make_record()
        with patch.object(scoring, "_DEEPPURPOSE_AVAILABLE", False):
            result = scoring.compute_reward(record)
        assert isinstance(result, RewardResult)


# ---------------------------------------------------------------------------
# 5. target_config sanity checks
# ---------------------------------------------------------------------------

class TestTargetConfig:
    def test_target_id_is_bace1_uniprot(self):
        from target.target_config import TARGET_ID
        assert TARGET_ID == "P56817"

    def test_model_name_correct(self):
        from target.target_config import MODEL_NAME
        assert MODEL_NAME == "MPNN_CNN_BindingDB"
        assert "Kd" not in MODEL_NAME

    def test_affinity_range_sensible(self):
        from target.target_config import AFFINITY_MAX, AFFINITY_MIN
        assert AFFINITY_MIN == pytest.approx(4.0)
        assert AFFINITY_MAX == pytest.approx(10.0)
        assert AFFINITY_MAX > AFFINITY_MIN

    def test_sequence_length(self):
        from target.target_config import TARGET_SEQUENCE
        assert len(TARGET_SEQUENCE) == 501, (
            f"TARGET_SEQUENCE length {len(TARGET_SEQUENCE)} ≠ 501 (UniProt P56817)"
        )

    def test_sequence_is_uppercase_amino_acids(self):
        from target.target_config import TARGET_SEQUENCE
        assert TARGET_SEQUENCE.isupper()
        assert TARGET_SEQUENCE.isalpha()