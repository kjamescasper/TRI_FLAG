"""
reporting/scoring.py

Implements compute_reward(record) -> RewardResult.

The reward is the number ACEGEN trains on. It must be:
  - In [0.0, 1.0]          (ACEGEN expects a normalised float)
  - Multiplicative          (all three properties must be simultaneously good;
                             additive formulas allow compensation)
  - Smooth                  (no binary cliff at 0.85 — ACEGEN needs a gradient)
  - Standalone              (called from run_record.py AND loop/triflag_scorer.py)

Formula (Week 8):
    reward = S_sa × S_nov × S_qed

    S_sa  — sigmoid-normalised SA score        (steepness k=1.5, midpoint=4.5)
    S_nov — two-zone Tanimoto novelty score    (hard zero ≥0.95, linear 0.70–0.95,
                                                full reward <0.70)
    S_qed — RDKit QED drug-likeness            (practical ceiling ~0.75)

Formula (Week 11):
    reward = S_sa × S_nov × S_qed × S_act

    S_act — normalised predicted pIC50 from DeepPurpose for BACE1.
            Defaults to 1.0 when DeepPurpose is not configured — formula
            degrades cleanly to the three-component Week 8 version.

Usage:
    from reporting.scoring import compute_reward
    result = compute_reward(record)
    record.reward = result.reward
    record.s_sa   = result.s_sa
    record.s_nov  = result.s_nov
    record.s_qed  = result.s_qed
    record.s_act  = result.s_act
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Week 11: attempt to import DeepPurpose target binding module.
# If the target package is absent (e.g. clean install without Week 11 files),
# _DEEPPURPOSE_AVAILABLE is False and S_act defaults to 1.0 in compute_reward().
# This ensures the formula degrades cleanly to the three-component version
# with zero performance overhead — no try/except in the hot path.
# ---------------------------------------------------------------------------

_DEEPPURPOSE_AVAILABLE: bool = False
predict_binding = None   # type: ignore[assignment]
normalise_affinity = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# S_act is gated by an explicit env var for two reasons:
#
# 1. Safety: torch fires a fatal C-level abort (0xc0000139, shm.dll) on
#    Windows when imported outside the acegen_scripts/run_gen0.py launcher
#    that patches the DLL search path.  A Python except clause cannot catch
#    a process-level abort, so we must never import DeepPurpose/torch at
#    module load time in normal pytest runs.
#
# 2. Performance: DeepPurpose model loading takes 5-10 seconds and should
#    only happen when ACEGEN is actually running.
#
# Set TRIFLAG_ENABLE_DEEPPURPOSE=1 in the environment (done automatically
# by acegen_scripts/run_gen0.py) to activate S_act.
# ---------------------------------------------------------------------------
import os as _os
if _os.environ.get("TRIFLAG_ENABLE_DEEPPURPOSE", "0") == "1":
    try:
        from target.deeppurpose_model import (  # type: ignore[import]
            normalise_affinity,
            predict_binding,
        )
        _DEEPPURPOSE_AVAILABLE = True
        logger.debug("DeepPurpose target binding module loaded — S_act active.")
    except Exception:  # noqa: BLE001
        logger.debug(
            "target.deeppurpose_model failed to load — S_act will default to 1.0."
        )
else:
    logger.debug(
        "TRIFLAG_ENABLE_DEEPPURPOSE not set — S_act disabled, "
        "formula degrades to three-component Week 10 version."
    )

# ---------------------------------------------------------------------------
# Tunable constants — centralised so tests can override and Week 9+ can tweak
# ---------------------------------------------------------------------------

# S_sa sigmoid parameters
SA_SIGMOID_MIDPOINT: float = 4.5   # SA score at which S_sa = 0.5
SA_SIGMOID_STEEPNESS: float = 1.5  # Controls sharpness of the sigmoid

# S_nov Tanimoto zone thresholds
NOV_HARD_CUTOFF: float = 0.95      # ≥ this → near-identical to prior art → S_nov = 0
NOV_BONUS_THRESHOLD: float = 0.70  # < this → genuinely novel scaffold → S_nov = 1.0
# Between NOV_BONUS_THRESHOLD and NOV_HARD_CUTOFF → linear gradient

# S_qed: computed from rdkit.Chem.QED — no thresholds needed


# ---------------------------------------------------------------------------
# RewardResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class RewardResult:
    """
    Structured output of compute_reward().

    All four components are stored in SQLite for dashboard decomposition.

    Attributes
    ----------
    reward : float
        Final reward in [0.0, 1.0].
    s_sa : float
        Sigmoid-normalised synthesisability component.
    s_nov : float
        Two-zone novelty component.
    s_qed : float
        RDKit QED drug-likeness component.
    s_act : float
        Target-binding component (Week 11). Defaults to 1.0 when DeepPurpose
        is unavailable so the multiplicative formula is unaffected.
    """

    reward: float
    s_sa: float
    s_nov: float
    s_qed: float
    s_act: float = 1.0

    def to_dict(self) -> dict:
        return {
            "reward": self.reward,
            "s_sa": self.s_sa,
            "s_nov": self.s_nov,
            "s_qed": self.s_qed,
            "s_act": self.s_act,
        }


# ---------------------------------------------------------------------------
# Component functions (public so loop/triflag_scorer.py can call directly)
# ---------------------------------------------------------------------------

def compute_s_sa(sa_score: float) -> float:
    """
    Sigmoid-normalised synthesisability score.

    Returns 1.0 for highly synthesisable molecules (SA≈1) and approaches
    0.0 for very complex molecules (SA≈10). S_sa = 0.5 when SA = SA_SIGMOID_MIDPOINT.

    Parameters
    ----------
    sa_score : float
        SA score in [1, 10].  Lower = more synthesisable.
    """
    # Logistic sigmoid: S_sa = 1 / (1 + exp(k * (sa - midpoint)))
    # Positive k, centred at midpoint: SA above midpoint → S_sa < 0.5
    exponent = SA_SIGMOID_STEEPNESS * (sa_score - SA_SIGMOID_MIDPOINT)
    s_sa = 1.0 / (1.0 + math.exp(exponent))
    return float(s_sa)


def compute_s_nov(nn_tanimoto: Optional[float]) -> float:
    """
    Two-zone Tanimoto novelty score.

    Zone 1: tanimoto >= NOV_HARD_CUTOFF → near-identical → S_nov = 0.0
    Zone 2: tanimoto < NOV_BONUS_THRESHOLD → genuinely novel → S_nov = 1.0
    Zone 3: linear gradient between the two thresholds

    This gives ACEGEN a smooth signal rather than a binary cliff at 0.85.

    Parameters
    ----------
    nn_tanimoto : float | None
        Nearest-neighbour Tanimoto similarity. None means no known neighbours
        were found — treated as perfectly novel (S_nov = 1.0).
    """
    if nn_tanimoto is None:
        return 1.0  # No known neighbours → assume novel

    t = float(nn_tanimoto)

    if t >= NOV_HARD_CUTOFF:
        return 0.0  # Near-identical to prior art

    if t < NOV_BONUS_THRESHOLD:
        return 1.0  # Genuinely novel scaffold

    # Linear gradient in [NOV_BONUS_THRESHOLD, NOV_HARD_CUTOFF)
    # t = NOV_BONUS_THRESHOLD → S_nov = 1.0
    # t → NOV_HARD_CUTOFF    → S_nov → 0.0
    gradient_range = NOV_HARD_CUTOFF - NOV_BONUS_THRESHOLD
    s_nov = 1.0 - (t - NOV_BONUS_THRESHOLD) / gradient_range
    return float(max(0.0, min(1.0, s_nov)))


def compute_s_qed(smiles: Optional[str]) -> float:
    """
    RDKit QED drug-likeness score.

    Returns a float in (0, 1]. Practical ceiling ~0.75 for drug-like molecules.
    Returns 0.5 as a neutral fallback if RDKit is unavailable or SMILES is None.

    Parameters
    ----------
    smiles : str | None
        Canonical SMILES string.
    """
    if not smiles:
        return 0.5

    try:
        from rdkit import Chem
        from rdkit.Chem import QED as RDKitQED

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.5
        return float(RDKitQED.qed(mol))
    except ImportError:
        logger.warning("RDKit not available — S_qed defaulting to 0.5")
        return 0.5
    except Exception as exc:
        logger.warning("QED calculation failed for %s: %s", smiles, exc)
        return 0.5


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_reward(record) -> RewardResult:
    """
    Compute the multiplicative four-component reward for a RunRecord.

    Returns RewardResult(reward=0.0, ...) immediately for:
      - invalid molecules (is_valid is False)
      - DISCARD decisions

    Week 11: S_act (DeepPurpose predicted pIC50) is multiplied in when
    target.deeppurpose_model is available. When absent, S_act=1.0 and the
    formula is identical to the three-component Week 8 version.

    Parameters
    ----------
    record : RunRecord
        The completed triage record. Reads: is_valid, final_decision,
        sa_score, nn_tanimoto, canonical_smiles (or smiles).
    """
    # --- Guard: invalid or discarded molecules get zero reward ---
    if not getattr(record, "is_valid", False):
        logger.debug("reward=0.0 — molecule is invalid")
        return RewardResult(reward=0.0, s_sa=0.0, s_nov=0.0, s_qed=0.0, s_act=1.0)

    if getattr(record, "final_decision", None) == "DISCARD":
        logger.debug("reward=0.0 — final_decision=DISCARD")
        return RewardResult(reward=0.0, s_sa=0.0, s_nov=0.0, s_qed=0.0, s_act=1.0)

    # --- Compute components ---
    sa_score = getattr(record, "sa_score", None)
    if sa_score is None:
        logger.warning("sa_score is None — defaulting S_sa to 0.5")
        s_sa = 0.5
    else:
        s_sa = compute_s_sa(float(sa_score))

    nn_tanimoto = getattr(record, "nn_tanimoto", None)
    s_nov = compute_s_nov(nn_tanimoto)

    smiles = (
        getattr(record, "smiles_canonical", None)
        or getattr(record, "canonical_smiles", None)
        or getattr(record, "smiles", None)
    )
    s_qed = compute_s_qed(smiles)

    # --- Week 11: S_act (DeepPurpose binding affinity) ---
    s_act: float = 1.0  # neutral default — preserves three-component formula
    if _DEEPPURPOSE_AVAILABLE and predict_binding is not None and normalise_affinity is not None:
        try:
            raw_pic50 = predict_binding(smiles or "")
            s_act = normalise_affinity(raw_pic50)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "S_act computation failed for SMILES %r: %s — defaulting to 1.0",
                (smiles or "")[:60],
                exc,
            )
            s_act = 1.0

    # --- Multiplicative combination (four components) ---
    reward = s_sa * s_nov * s_qed * s_act

    logger.debug(
        "reward=%.4f  s_sa=%.4f  s_nov=%.4f  s_qed=%.4f  s_act=%.4f"
        "  (sa=%.2f, tanimoto=%s)",
        reward,
        s_sa,
        s_nov,
        s_qed,
        s_act,
        sa_score if sa_score is not None else -1,
        f"{nn_tanimoto:.3f}" if nn_tanimoto is not None else "None",
    )

    return RewardResult(reward=reward, s_sa=s_sa, s_nov=s_nov, s_qed=s_qed, s_act=s_act)