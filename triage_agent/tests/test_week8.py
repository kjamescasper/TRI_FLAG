"""
tests/test_week8.py

Week 8 test suite.

Covers:
  1. reward = 0.0 for invalid molecules
  2. reward = 0.0 for DISCARD decisions
  3. Known drugs with high Tanimoto (≥ 0.95) return S_nov = 0 → reward = 0
  4. S_nov zone transitions fire correctly at 0.70 and 0.95
  5. S_sa sigmoid value at SA=4.5 equals exactly 0.5
  6. SQLite save/load round-trip preserves all key fields
  7. get_top_n_by_reward returns rows ordered by reward DESC
  8. save_batch upserts correctly

No network calls. All tests use in-memory SQLite and mock RunRecord objects.

Run with:
    set PYTHONPATH=. && pytest tests/test_week8.py -v
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pytest

from reporting.scoring import (
    RewardResult,
    compute_reward,
    compute_s_nov,
    compute_s_sa,
    NOV_BONUS_THRESHOLD,
    NOV_HARD_CUTOFF,
    SA_SIGMOID_MIDPOINT,
)
from database.db import DatabaseManager


# ===========================================================================
# Minimal mock RunRecord
# ===========================================================================

@dataclass
class MockRunRecord:
    """Lightweight stand-in for RunRecord — no pipeline dependencies."""

    molecule_id: str = field(default_factory=lambda: f"mol_{uuid.uuid4().hex[:8]}")
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:8]}")
    smiles: str = "CCO"
    canonical_smiles: str = "CCO"
    is_valid: bool = True
    final_decision: str = "PASS"
    sa_score: Optional[float] = 2.5
    nn_tanimoto: Optional[float] = 0.30
    triaged_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    batch_id: Optional[str] = None
    generation_number: Optional[int] = None
    entry_point: Optional[str] = "cli"
    reward: Optional[float] = None
    s_sa: Optional[float] = None
    s_nov: Optional[float] = None
    s_qed: Optional[float] = None
    s_act: Optional[float] = None
    inchi: Optional[str] = None
    inchikey: Optional[str] = None
    molecular_formula: Optional[str] = None
    source: str = "triflag"
    rationale: Optional[str] = None
    sa_decision: Optional[str] = None
    sa_category: Optional[str] = None
    nn_source: Optional[str] = None
    nn_id: Optional[str] = None
    nn_name: Optional[str] = None
    similarity_decision: Optional[str] = None
    validity_error: Optional[str] = None
    mol_weight: Optional[float] = None
    logp: Optional[float] = None
    tpsa: Optional[float] = None
    hbd: Optional[int] = None
    hba: Optional[int] = None
    rotatable_bonds: Optional[int] = None
    scaffold_smiles: Optional[str] = None
    pains_alert: Optional[bool] = None
    predicted_affinity: Optional[float] = None
    target_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ===========================================================================
# Helper: in-memory DatabaseManager
# ===========================================================================

@pytest.fixture()
def db(tmp_path):
    """Fresh DatabaseManager backed by a temp file (avoids :memory: issues
    with foreign keys across connections)."""
    return DatabaseManager(db_path=str(tmp_path / "test.db"))


def _make_record(**kwargs) -> MockRunRecord:
    return MockRunRecord(**kwargs)


# ===========================================================================
# 1. Invalid molecules → reward = 0
# ===========================================================================

class TestRewardInvalidMolecule:

    def test_invalid_molecule_returns_zero(self):
        record = _make_record(is_valid=False, final_decision="DISCARD")
        result = compute_reward(record)
        assert result.reward == 0.0

    def test_invalid_molecule_all_components_zero(self):
        record = _make_record(is_valid=False)
        result = compute_reward(record)
        assert result.s_sa == 0.0
        assert result.s_nov == 0.0
        assert result.s_qed == 0.0

    def test_invalid_molecule_is_reward_result(self):
        record = _make_record(is_valid=False)
        result = compute_reward(record)
        assert isinstance(result, RewardResult)


# ===========================================================================
# 2. DISCARD decisions → reward = 0
# ===========================================================================

class TestRewardDiscardDecision:

    def test_discard_valid_molecule_returns_zero(self):
        record = _make_record(is_valid=True, final_decision="DISCARD", sa_score=3.0)
        result = compute_reward(record)
        assert result.reward == 0.0

    def test_discard_with_low_sa_still_zero(self):
        """Even a very easy-to-synthesise molecule that was DISCARDed gets 0."""
        record = _make_record(
            is_valid=True,
            final_decision="DISCARD",
            sa_score=1.5,
            nn_tanimoto=0.10,
        )
        result = compute_reward(record)
        assert result.reward == 0.0


# ===========================================================================
# 3. High Tanimoto (≥ 0.95) → S_nov = 0 → reward = 0
# ===========================================================================

class TestNoveltyCollapse:

    def test_tanimoto_095_kills_reward(self):
        record = _make_record(
            is_valid=True,
            final_decision="PASS",
            sa_score=2.0,
            nn_tanimoto=0.95,
        )
        result = compute_reward(record)
        assert result.reward == 0.0
        assert result.s_nov == 0.0

    def test_tanimoto_exactly_1_kills_reward(self):
        record = _make_record(
            is_valid=True,
            final_decision="PASS",
            sa_score=1.5,
            nn_tanimoto=1.0,
        )
        result = compute_reward(record)
        assert result.reward == 0.0
        assert result.s_nov == 0.0

    def test_tanimoto_just_below_095_nonzero_reward(self):
        """0.949 is in the gradient zone — reward should be positive."""
        record = _make_record(
            is_valid=True,
            final_decision="PASS",
            sa_score=2.0,
            nn_tanimoto=0.949,
        )
        result = compute_reward(record)
        assert result.s_nov > 0.0
        assert result.reward > 0.0


# ===========================================================================
# 4. S_nov zone transitions at 0.70 and 0.95
# ===========================================================================

class TestSNovZones:

    def test_below_bonus_threshold_is_1(self):
        """Tanimoto < 0.70 → S_nov = 1.0 (fully novel)."""
        assert compute_s_nov(0.0) == 1.0
        assert compute_s_nov(0.50) == 1.0
        assert compute_s_nov(NOV_BONUS_THRESHOLD - 0.001) == 1.0

    def test_at_bonus_threshold_is_1(self):
        """Exactly at lower threshold → S_nov = 1.0."""
        assert compute_s_nov(NOV_BONUS_THRESHOLD) == 1.0

    def test_at_hard_cutoff_is_0(self):
        """Exactly at hard cutoff → S_nov = 0.0."""
        assert compute_s_nov(NOV_HARD_CUTOFF) == 0.0

    def test_gradient_zone_is_interpolated(self):
        """Midpoint of gradient zone → S_nov ≈ 0.5."""
        mid = (NOV_BONUS_THRESHOLD + NOV_HARD_CUTOFF) / 2
        s_nov = compute_s_nov(mid)
        assert 0.4 < s_nov < 0.6

    def test_none_tanimoto_is_fully_novel(self):
        """No nearest neighbour found → S_nov = 1.0."""
        assert compute_s_nov(None) == 1.0

    def test_gradient_is_monotonically_decreasing(self):
        """As Tanimoto increases from 0.70 to 0.95, S_nov must decrease."""
        values = [
            compute_s_nov(t)
            for t in [0.70, 0.75, 0.80, 0.85, 0.90, 0.94]
        ]
        for earlier, later in zip(values, values[1:]):
            assert earlier >= later, f"S_nov not monotone: {earlier} < {later}"


# ===========================================================================
# 5. S_sa sigmoid: value at SA=4.5 equals 0.5
# ===========================================================================

class TestSSaSigmoid:

    def test_midpoint_returns_half(self):
        s_sa = compute_s_sa(SA_SIGMOID_MIDPOINT)
        assert abs(s_sa - 0.5) < 1e-9

    def test_low_sa_near_1(self):
        """SA=1 (very easy) → S_sa close to 1.0."""
        assert compute_s_sa(1.0) > 0.85

    def test_high_sa_near_0(self):
        """SA=10 (very hard) → S_sa close to 0.0."""
        assert compute_s_sa(10.0) < 0.15

    def test_sigmoid_is_monotonically_decreasing(self):
        """Higher SA score → lower S_sa."""
        scores = [1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0]
        s_values = [compute_s_sa(s) for s in scores]
        for a, b in zip(s_values, s_values[1:]):
            assert a > b


# ===========================================================================
# 6. SQLite round-trip: save_run then load preserves key fields
# ===========================================================================

class TestSQLiteRoundTrip:

    def test_save_and_retrieve_by_batch(self, db):
        record = _make_record(
            batch_id="batch_gen1",
            generation_number=1,
            final_decision="PASS",
            sa_score=3.2,
            nn_tanimoto=0.40,
            reward=0.72,
            s_sa=0.85,
            s_nov=0.88,
            s_qed=0.96,
        )
        db.save_run(record)
        rows = db.load_runs_by_batch("batch_gen1")
        assert len(rows) == 1
        row = rows[0]
        assert row["run_id"] == record.run_id
        assert row["molecule_id"] == record.molecule_id
        assert row["final_decision"] == "PASS"
        assert abs(row["sa_score"] - 3.2) < 1e-6
        assert abs(row["nn_tanimoto"] - 0.40) < 1e-6

    def test_reward_persisted(self, db):
        record = _make_record(reward=0.618, s_sa=0.9, s_nov=0.8, s_qed=0.86)
        db.save_run(record)
        row = db.get_run_by_id(record.run_id)
        assert row is not None
        assert abs(row["reward"] - 0.618) < 1e-6

    def test_is_valid_stored_as_integer(self, db):
        record = _make_record(is_valid=True)
        db.save_run(record)
        row = db.get_run_by_id(record.run_id)
        assert row["is_valid"] == 1

    def test_nullable_fields_are_none_by_default(self, db):
        record = _make_record()
        db.save_run(record)
        row = db.get_run_by_id(record.run_id)
        assert row["logp"] is None
        assert row["tpsa"] is None
        assert row["predicted_affinity"] is None

    def test_duplicate_molecule_id_no_crash(self, db):
        """Same molecule triaged twice → second INSERT OR IGNORE on molecules."""
        mol_id = "mol_shared"
        r1 = _make_record(molecule_id=mol_id, final_decision="PASS")
        r2 = _make_record(molecule_id=mol_id, final_decision="FLAG")
        db.save_run(r1)
        db.save_run(r2)  # Should not raise
        rows = db.get_all_runs()
        assert len(rows) == 2


# ===========================================================================
# 7. get_top_n_by_reward ordering
# ===========================================================================

class TestTopNByReward:

    def test_returns_highest_reward_first(self, db):
        rewards = [0.30, 0.90, 0.55, 0.10, 0.75]
        for r in rewards:
            rec = _make_record(
                batch_id="batch_rank",
                final_decision="PASS",
                reward=r,
            )
            db.save_run(rec)

        top3 = db.get_top_n_by_reward(n=3, batch_id="batch_rank")
        reward_values = [row["reward"] for row in top3]
        assert reward_values == sorted(reward_values, reverse=True)
        assert reward_values[0] == pytest.approx(0.90)

    def test_top_n_respects_limit(self, db):
        for _ in range(5):
            db.save_run(_make_record(batch_id="batch_limit", reward=0.5))
        top2 = db.get_top_n_by_reward(n=2, batch_id="batch_limit")
        assert len(top2) == 2

    def test_top_n_without_batch_crosses_all(self, db):
        db.save_run(_make_record(batch_id="a", reward=0.99))
        db.save_run(_make_record(batch_id="b", reward=0.01))
        top = db.get_top_n_by_reward(n=1)
        assert top[0]["reward"] == pytest.approx(0.99)


# ===========================================================================
# 8. save_batch upsert
# ===========================================================================

class TestSaveBatch:

    def test_save_batch_no_crash(self, db):
        db.save_batch("gen_001", {
            "generation_number": 1,
            "source": "acegen",
            "molecule_count": 1000,
            "pass_count": 200,
            "flag_count": 150,
            "discard_count": 650,
            "mean_reward": 0.42,
        })
        # Verify it's in the DB
        conn = db._conn()
        row = conn.execute(
            "SELECT * FROM batches WHERE batch_id = ?", ("gen_001",)
        ).fetchone()
        assert row is not None
        assert row["molecule_count"] == 1000
        assert abs(row["mean_reward"] - 0.42) < 1e-6

    def test_save_batch_upsert(self, db):
        """Saving the same batch_id twice updates, not duplicates."""
        db.save_batch("gen_dup", {"molecule_count": 100})
        db.save_batch("gen_dup", {"molecule_count": 200})
        conn = db._conn()
        rows = conn.execute(
            "SELECT * FROM batches WHERE batch_id = ?", ("gen_dup",)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["molecule_count"] == 200


# ===========================================================================
# 9. Reward is multiplicative (not additive)
# ===========================================================================

class TestMultiplicativeProperty:

    def test_zero_s_nov_kills_entire_reward(self):
        """If S_nov = 0 (tanimoto ≥ 0.95), reward must be 0 regardless of SA."""
        record = _make_record(
            is_valid=True,
            final_decision="PASS",
            sa_score=1.0,   # S_sa ≈ 0.99 — excellent
            nn_tanimoto=0.99,  # S_nov = 0
            canonical_smiles="CCO",
        )
        result = compute_reward(record)
        assert result.reward == 0.0

    def test_perfect_components_give_high_reward(self):
        """SA=1, tanimoto=0, novel SMILES → high reward."""
        record = _make_record(
            is_valid=True,
            final_decision="PASS",
            sa_score=1.0,
            nn_tanimoto=0.0,
            canonical_smiles="CCO",
        )
        result = compute_reward(record)
        # S_sa(1.0) > 0.85, S_nov = 1.0 → reward > 0.5 with any reasonable S_qed
        assert result.reward > 0.3