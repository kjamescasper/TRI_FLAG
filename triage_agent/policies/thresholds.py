"""
policies/thresholds.py

Threshold definitions for TRI_FLAG molecular triage system.
All values are literature-justified and centrally managed here.

Two complementary classification systems:

1. SAScoreCategoryThresholds — 4-tier descriptive label
   (easy / moderate / difficult / very_difficult). Used for explainability.

2. SAScoreThresholds — 3-tier pipeline decision (PASS / FLAG / DISCARD).
   Used by PolicyEngine to route molecules through the pipeline.

These are deliberately separate: a molecule can be "moderate" in category
terms but still PASS the pipeline filter, keeping the systems independently
tunable.

References:
    [1] Ertl, P. & Schuffenhauer, A. (2009). Estimation of synthetic
        accessibility score of drug-like molecules based on molecular
        complexity and fragment contributions. J. Cheminform., 1, 8.
        DOI: 10.1186/1758-2946-1-8

    [2] Bickerton, G.R. et al. (2012). Quantifying the chemical beauty
        of drugs. Nature Chemistry, 4, 90-98.
        DOI: 10.1038/nchem.1243

    [3] Gao, W. & Coley, C.W. (2020). The Synthesizability of Molecules
        Proposed by Generative Models. J. Chem. Inf. Model., 60, 5714-5723.
        DOI: 10.1021/acs.jcim.0c00174

    [4] Lovering, F. et al. (2009). Escape from flatland: increasing
        saturation as an approach to improving clinical success.
        J. Med. Chem., 52, 6752-6756.
        DOI: 10.1021/jm901241e
"""

from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# Type aliases (used for annotation only — not runtime-enforced)
# ---------------------------------------------------------------------------

CategoryLabel = str   # "easy" | "moderate" | "difficult" | "very_difficult"
PipelineDecision = str  # "PASS" | "FLAG" | "DISCARD"


# =============================================================================
# System 1: Descriptive Categorization (4-tier)
# =============================================================================
#
# Based on Ertl & Schuffenhauer [1] and validated by Gao & Coley [3]:
#
#   easy          SA ≤ 3.0 — Simple, commercially available building blocks.
#                            Aspirin = ~1.7, Ibuprofen = ~2.1 [1]
#   moderate      SA ≤ 6.0 — Typical drug candidates. Most marketed small
#                            molecules fall in the 2-5 range [2].
#   difficult     SA ≤ 7.0 — Natural product-like. Specialist synthesis needed.
#   very_difficult SA > 7.0 — Rarely synthesised; generative model outputs at
#                            this score are typically infeasible [3].

class SAScoreCategoryThresholds:
    """
    4-tier descriptive synthesis difficulty classification.

    These are reporting thresholds. They do NOT directly control pipeline
    routing — use SAScoreThresholds for pipeline decisions.
    """

    def __init__(
        self,
        easy_max: float = 3.0,
        moderate_max: float = 6.0,
        difficult_max: float = 7.0,
    ):
        self.easy_max = easy_max
        self.moderate_max = moderate_max
        self.difficult_max = difficult_max

    def categorize(self, sa_score: float) -> CategoryLabel:
        """Return the 4-tier synthesis difficulty label for an SA score."""
        if sa_score <= self.easy_max:
            return "easy"
        elif sa_score <= self.moderate_max:
            return "moderate"
        elif sa_score <= self.difficult_max:
            return "difficult"
        else:
            return "very_difficult"

    def to_dict(self) -> Dict[str, float]:
        return {
            "easy_max": self.easy_max,
            "moderate_max": self.moderate_max,
            "difficult_max": self.difficult_max,
        }


# =============================================================================
# System 2: Pipeline Decision Thresholds (3-tier)
# =============================================================================
#
#   PASS    SA < pass_threshold (6.0)
#           Straightforward synthesis; proceed to next filter.
#           Rationale: ~95% of known drugs score ≤ 6 [1].
#
#   FLAG    pass_threshold ≤ SA ≤ flag_threshold (6.0–7.0)
#           Challenging but feasible; log warning, continue pipeline.
#           Rationale: borderline cases benefit from human review [2].
#
#   DISCARD SA > flag_threshold (7.0)
#           Synthetically intractable for drug development.
#           Rationale: SA > 7 strongly predicts infeasibility in
#           generative model outputs [3].

class SAScoreThresholds:
    """
    3-tier pipeline routing thresholds with embedded 4-tier categorization.

    Usage:
        t = SAScoreThresholds()
        decision  = t.classify(sa_score)   # -> "PASS" | "FLAG" | "DISCARD"
        category  = t.categorize(sa_score) # -> "easy" | "moderate" | ...
        desc      = t.describe(sa_score)   # -> human-readable string

    To customise:
        strict = SAScoreThresholds(pass_threshold=5.0, flag_threshold=6.0)
    """

    def __init__(
        self,
        pass_threshold: float = 6.0,
        flag_threshold: float = 7.0,
        score_min: float = 1.0,
        score_max: float = 10.0,
        category_thresholds: SAScoreCategoryThresholds = None,
    ):
        self.pass_threshold = pass_threshold
        self.flag_threshold = flag_threshold
        self.score_min = score_min
        self.score_max = score_max
        # Allow custom category thresholds; default aligns with pipeline thresholds
        self._category = category_thresholds or SAScoreCategoryThresholds()

    def classify(self, sa_score: float) -> PipelineDecision:
        """Return the 3-tier pipeline routing decision for an SA score."""
        if sa_score < self.pass_threshold:
            return "PASS"
        elif sa_score <= self.flag_threshold:
            return "FLAG"
        else:
            return "DISCARD"

    def categorize(self, sa_score: float) -> CategoryLabel:
        """Return the 4-tier descriptive synthesis difficulty category."""
        return self._category.categorize(sa_score)

    def is_valid_range(self, sa_score: float) -> bool:
        """Return True if score is within the expected theoretical range [1, 10]."""
        return self.score_min <= sa_score <= self.score_max

    def describe(self, sa_score: float) -> str:
        """Return a human-readable combined description of decision + category."""
        decision = self.classify(sa_score)
        category = self.categorize(sa_score)

        if decision == "PASS":
            return (
                f"SA score {sa_score:.2f} ({category}) — straightforward synthesis; "
                f"below FLAG threshold of {self.pass_threshold}."
            )
        elif decision == "FLAG":
            return (
                f"SA score {sa_score:.2f} ({category}) — challenging synthesis; "
                f"in FLAG range [{self.pass_threshold}, {self.flag_threshold}]. "
                "Pipeline continues with warning."
            )
        else:  # DISCARD
            return (
                f"SA score {sa_score:.2f} ({category}) — synthetically intractable; "
                f"exceeds DISCARD threshold of {self.flag_threshold}."
            )

    def to_dict(self) -> Dict[str, float]:
        return {
            "pass_threshold": self.pass_threshold,
            "flag_threshold": self.flag_threshold,
            "score_min": self.score_min,
            "score_max": self.score_max,
        }


# =============================================================================
# Pre-configured threshold sets for different project contexts
# =============================================================================

DEFAULT_SA_THRESHOLDS = SAScoreThresholds(
    pass_threshold=6.0,
    flag_threshold=7.0,
)
"""
Standard thresholds for general drug discovery triage.
Suitable for hit identification and hit-to-lead campaigns.
"""

LEAD_OPTIMIZATION_THRESHOLDS = SAScoreThresholds(
    pass_threshold=5.0,
    flag_threshold=6.0,
    category_thresholds=SAScoreCategoryThresholds(
        easy_max=2.5, moderate_max=5.0, difficult_max=6.0
    ),
)
"""
Stricter thresholds for lead optimization.
At this stage each analog must be readily synthesisable by a medicinal
chemistry team. Lead compounds should ideally score < 4 [2].
"""

NATURAL_PRODUCT_THRESHOLDS = SAScoreThresholds(
    pass_threshold=7.0,
    flag_threshold=9.0,
    category_thresholds=SAScoreCategoryThresholds(
        easy_max=4.0, moderate_max=7.0, difficult_max=9.0
    ),
)
"""
Permissive thresholds for natural product-inspired campaigns.
Natural products routinely score 6-9; standard thresholds would
incorrectly discard most of the relevant chemical space.
"""

FRAGMENT_SCREENING_THRESHOLDS = SAScoreThresholds(
    pass_threshold=3.5,
    flag_threshold=5.0,
    category_thresholds=SAScoreCategoryThresholds(
        easy_max=2.0, moderate_max=3.5, difficult_max=5.0
    ),
)
"""
Very strict thresholds for fragment-based drug discovery.
Fragment libraries contain only the simplest building blocks (SA < 3)
to allow rapid elaboration.
"""


# =============================================================================
# Validity atom count limits (centralised from Week 3)
# =============================================================================

VALIDITY_MIN_ATOMS: int = 1
"""Minimum heavy atom count for a molecule to be considered valid."""

VALIDITY_MAX_ATOMS: int = 500
"""
Maximum heavy atom count before flagging as anomalously large.
Typical drug-like molecules have 20-70 heavy atoms. 500 is a generous
ceiling to catch degenerate generative model outputs.
"""