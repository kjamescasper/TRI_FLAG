# tools/pains_tool.py
"""
PAINSTool — Week 9

Screens molecules for Pan-Assay Interference Compounds (PAINS) structural alerts
using RDKit's built-in FilterCatalog with the PAINS_A/B/C pattern sets.

PAINS are motifs known to produce false positives in biochemical assays due to
reactivity, fluorescence interference, or aggregation — not due to genuine binding.

Non-terminal: a PAINS match generates an advisory FLAG with pattern names recorded
in the rationale, but never causes DISCARD. PAINS is a red flag for wet-lab
follow-up, not a disqualifier.

Position in pipeline: after SimilarityTool (final tool in the Week 9 pipeline).
"""

import logging
import time
from typing import Any, Dict, List

from tools.base_tool import Tool
from agent.agent_state import AgentState

try:
    from rdkit import Chem
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Build the PAINS filter catalog once at module load — it's expensive to construct.
_PAINS_CATALOG: "FilterCatalog | None" = None

def _get_pains_catalog() -> "FilterCatalog | None":
    """Return a singleton PAINS FilterCatalog, building it on first call."""
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None and RDKIT_AVAILABLE:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
        _PAINS_CATALOG = FilterCatalog(params)
    return _PAINS_CATALOG


class PAINSTool(Tool):
    """
    Screen for PAINS structural alerts using RDKit FilterCatalog.

    Reads canonical SMILES from ValidityTool result. Stores results in
    state.tool_results['PAINSTool']. A match generates a FLAG but never
    terminates the pipeline.
    """

    def __init__(self):
        self.name = "PAINSTool"
        self.description = (
            "Screens for PAINS structural alerts (Pan-Assay Interference Compounds) "
            "using RDKit FilterCatalog with PAINS_A/B/C pattern sets."
        )
        self.logger = logging.getLogger(__name__)

        if not RDKIT_AVAILABLE:
            self.logger.error(
                "RDKit not available — PAINSTool will skip all molecules."
            )

    def run(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute PAINS screening.

        Args:
            state: AgentState after ValidityTool has run.

        Returns:
            Dict with keys: pains_alert (bool), pains_matches (list[str]),
            error_message (str|None), execution_time_ms (float).
        """
        t0 = time.perf_counter()

        null_result = {
            "tool_name": self.name,
            "molecule_id": state.molecule_id,
            "pains_alert": False,
            "pains_matches": [],
            "error_message": None,
            "execution_time_ms": 0.0,
        }

        if not RDKIT_AVAILABLE:
            null_result["error_message"] = "RDKit not available"
            return null_result

        # --- Guard: validity must have passed ---
        validity = state.tool_results.get("ValidityTool", {})
        if validity and not validity.get("is_valid", True):
            null_result["error_message"] = "Molecule invalid — PAINS skipped"
            return null_result

        # --- Extract SMILES ---
        smiles = None
        if validity:
            smiles = validity.get("smiles_canonical") or validity.get("smiles")
        if not smiles:
            smiles = state.raw_input if isinstance(state.raw_input, str) else None

        if not smiles:
            null_result["error_message"] = "No SMILES available"
            return null_result

        # --- Build mol ---
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            null_result["error_message"] = f"RDKit could not parse: {smiles!r}"
            return null_result

        # --- Run PAINS filter ---
        try:
            catalog = _get_pains_catalog()
            if catalog is None:
                null_result["error_message"] = "PAINS catalog unavailable"
                return null_result

            matches: List[str] = []
            entries = catalog.GetMatches(mol)
            for entry in entries:
                matches.append(entry.GetDescription())

            pains_alert = len(matches) > 0

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.logger.warning("PAINSTool error for %s: %s", smiles, exc)
            null_result["error_message"] = str(exc)
            null_result["execution_time_ms"] = round(elapsed_ms, 2)
            return null_result

        # --- Generate advisory FLAG if PAINS matched ---
        if pains_alert:
            pattern_list = ", ".join(matches[:5])  # cap display at 5
            state.add_flag(
                reason=f"PAINS structural alert: {pattern_list}",
                source=self.name,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "tool_name": self.name,
            "molecule_id": state.molecule_id,
            "pains_alert": pains_alert,
            "pains_matches": matches,
            "error_message": None,
            "execution_time_ms": round(elapsed_ms, 2),
        }