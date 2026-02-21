"""
tools/similarity_tool.py

Similarity-based IP-risk screening via ChEMBL and PubChem REST APIs.

Queries two public cheminformatics databases at runtime:
  - ChEMBL REST API (~2.4M compounds): the gold standard for drug-like
    bioactive chemistry. Returns Tanimoto scores computed with RDKit Morgan
    fingerprints internally — directly comparable to our local ECFP4.
  - PubChem PUG REST API (~115M compounds): broadest possible structure
    coverage including industrial chemicals, patents, research compounds.
    Uses PubChem's own 881-bit fingerprint; scores are proxies only.

Design principles:
  - FLAG-only (never DISCARD) — similarity is a proxy, not a legal judgment.
    A chemist / patent attorney must make the final IP determination.
  - FLAG-and-continue — consistent with SAScoreTool behavior.
  - All API failures are non-terminal. If both APIs fail, ERROR is returned
    and the PolicyEngine flags conservatively.
  - SimilarityTool always runs, even if SAScoreTool flagged the molecule.
    Rationale: synthetically challenging + novel is different from
    synthetically challenging + known drug.
  - SMILES are read from state.tool_results['ValidityTool']['smiles_canonical'].
    This avoids re-canonicalization and ensures consistency.

Performance note:
  python main.py will take ~3-8s with SimilarityTool registered due to API
  round trips. This is the expected trade-off for live database coverage.

Literature:
  Maggiora G et al. (2014). J. Med. Chem. 57(8), 3186-3204.
    DOI: 10.1021/jm401411z — ECFP4 Tanimoto is standard for IP-risk.
  Bender A, Glen RC (2004). Org. Biomol. Chem. 2, 3204-3218.
    DOI: 10.1039/b409813g — 0.85 threshold minimizes false positives.

Week: 5
"""

import logging
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

from tools.base_tool import Tool
from agent.agent_state import AgentState
from chemistry.fingerprints import (
    morgan_fingerprint_from_smiles,
    fingerprint_to_hex,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# requests import guard — should always be available, but guard for safety
# ---------------------------------------------------------------------------
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    logger.error(
        "requests library not available. SimilarityTool will fail gracefully. "
        "Install with: conda install -c conda-forge requests"
    )

# ---------------------------------------------------------------------------
# Optional chembl_webresource_client — preferred; falls back to raw requests
# ---------------------------------------------------------------------------
try:
    from chembl_webresource_client.new_client import new_client as _chembl_client
    _CHEMBL_CLIENT_AVAILABLE = True
    logger.debug("chembl_webresource_client available — using Python client for ChEMBL")
except ImportError:
    _CHEMBL_CLIENT_AVAILABLE = False
    logger.debug(
        "chembl_webresource_client not installed — using raw requests for ChEMBL. "
        "Install with: conda install -c conda-forge chembl_webresource_client"
    )

# ---------------------------------------------------------------------------
# API constants
# ---------------------------------------------------------------------------
_CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
_PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Fingerprint method tag stored in result for provenance
_FINGERPRINT_METHOD = "Morgan_ECFP4_r2_2048bits"


class SimilarityTool(Tool):
    """
    Chemical similarity screening tool using live public cheminformatics APIs.

    Compares the query molecule against ~2.4M ChEMBL compounds and ~115M
    PubChem compounds. Tanimoto >= flag_threshold triggers FLAG annotation on
    AgentState (pipeline continues — no early termination).

    Execution flow per molecule:
        1. Extract canonical SMILES from ValidityTool result in state
        2. Compute local Morgan ECFP4 fingerprint (for provenance)
        3. Query ChEMBL similarity API (if use_chembl=True)
        4. Query PubChem similarity API (if use_pubchem=True)
        5. Merge hits, find global nearest neighbor
        6. Apply threshold: >= flag_threshold -> FLAG, else PASS

    Result dict stored in state.tool_results['SimilarityTool']:
        {
            'tool_name': 'SimilarityTool',
            'molecule_id': str,
            'query_smiles': str,
            'similarity_decision': 'PASS' | 'FLAG' | 'ERROR',
            'nearest_neighbor_tanimoto': float,
            'nearest_neighbor_source': 'ChEMBL' | 'PubChem' | None,
            'nearest_neighbor_id': str | None,
            'nearest_neighbor_name': str | None,
            'nearest_neighbor_smiles': str | None,
            'chembl_hits': List[dict],
            'pubchem_hits': List[dict],
            'flag_threshold_used': float,
            'fingerprint_method': str,
            'query_fingerprint_hex': str | None,
            'apis_queried': List[str],
            'chembl_available': bool,
            'pubchem_available': bool,
            'execution_time_ms': float,
        }

    Error handling (all non-terminal):
        - API timeout -> mark that API as unavailable, continue with other
        - Both APIs fail -> similarity_decision='ERROR', no termination
        - SMILES not found in state -> ERROR result, logged as warning
    """

    name = "SimilarityTool"

    def __init__(
        self,
        flag_threshold: float = 0.85,
        use_chembl: bool = True,
        use_pubchem: bool = True,
        chembl_timeout: float = 10.0,
        pubchem_timeout: float = 15.0,
        max_results_per_api: int = 5,
    ):
        """
        Initialize the SimilarityTool.

        Args:
            flag_threshold: Tanimoto score >= this value triggers FLAG.
                            Default 0.85 (Maggiora 2014; Bender & Glen 2004).
            use_chembl: Whether to query the ChEMBL REST API.
            use_pubchem: Whether to query the PubChem PUG REST API.
            chembl_timeout: Seconds before ChEMBL request times out.
            pubchem_timeout: Seconds total for PubChem async polling.
            max_results_per_api: Maximum hits to retrieve from each API.
        """
        self.flag_threshold = flag_threshold
        self.use_chembl = use_chembl
        self.use_pubchem = use_pubchem
        self.chembl_timeout = chembl_timeout
        self.pubchem_timeout = pubchem_timeout
        self.max_results_per_api = max_results_per_api
        self.description = (
            "Similarity-based IP-risk screening via ChEMBL and PubChem APIs. "
            f"Flags molecules with Tanimoto >= {flag_threshold:.2f}."
        )
        logger.info(
            "SimilarityTool initialized: flag_threshold=%.2f, "
            "use_chembl=%s, use_pubchem=%s, max_results=%d",
            flag_threshold, use_chembl, use_pubchem, max_results_per_api,
        )

    # =========================================================================
    # Public entry point
    # =========================================================================

    def run(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute similarity screening for the molecule in state.

        Reads canonical SMILES from state.tool_results['ValidityTool']['smiles_canonical'].
        Always runs regardless of whether prior tools flagged the molecule.

        Args:
            state: AgentState with ValidityTool result already stored

        Returns:
            Result dictionary (see class docstring for full schema)
        """
        start_time = time.monotonic()
        molecule_id = state.molecule_id

        logger.info("SimilarityTool: starting for molecule_id=%s", molecule_id)

        # Step 1: Extract canonical SMILES from ValidityTool result
        smiles, smiles_error = self._extract_smiles(state)

        if smiles is None:
            logger.warning(
                "SimilarityTool [%s]: cannot extract SMILES — %s",
                molecule_id, smiles_error,
            )
            return self._error_result(
                molecule_id=molecule_id,
                query_smiles=None,
                error_reason=smiles_error or "No canonical SMILES available",
                execution_time_ms=self._elapsed_ms(start_time),
            )

        # Step 2: Compute local fingerprint for provenance
        local_fp = morgan_fingerprint_from_smiles(smiles)
        fp_hex = fingerprint_to_hex(local_fp)

        logger.debug(
            "SimilarityTool [%s]: canonical SMILES=%r, fp_hex=%s",
            molecule_id, smiles, fp_hex[:12] + "..." if fp_hex else "None",
        )

        # Step 3: Query APIs
        chembl_hits: List[dict] = []
        pubchem_hits: List[dict] = []
        chembl_available = False
        pubchem_available = False
        apis_queried: List[str] = []

        if self.use_chembl and _REQUESTS_AVAILABLE:
            chembl_hits, chembl_available = self._query_chembl(smiles, molecule_id)
            if chembl_available:
                apis_queried.append("ChEMBL")

        if self.use_pubchem and _REQUESTS_AVAILABLE:
            pubchem_hits, pubchem_available = self._query_pubchem(smiles, molecule_id)
            if pubchem_available:
                apis_queried.append("PubChem")

        if not _REQUESTS_AVAILABLE:
            logger.error(
                "SimilarityTool [%s]: requests library unavailable", molecule_id
            )

        # Step 4: Find global nearest neighbor across all hits
        nn_tanimoto, nn_source, nn_id, nn_name, nn_smiles = self._find_nearest_neighbor(
            chembl_hits=chembl_hits,
            pubchem_hits=pubchem_hits,
        )

        # Step 5: Apply threshold
        if not chembl_available and not pubchem_available:
            # Both APIs failed
            similarity_decision = "ERROR"
        elif nn_tanimoto >= self.flag_threshold:
            similarity_decision = "FLAG"
        else:
            similarity_decision = "PASS"

        execution_time_ms = self._elapsed_ms(start_time)

        logger.info(
            "SimilarityTool [%s]: decision=%s, nearest_neighbor=%.3f from %s",
            molecule_id, similarity_decision, nn_tanimoto, nn_source or "N/A",
        )

        return {
            "tool_name": self.name,
            "molecule_id": molecule_id,
            "query_smiles": smiles,
            "similarity_decision": similarity_decision,
            "nearest_neighbor_tanimoto": nn_tanimoto,
            "nearest_neighbor_source": nn_source,
            "nearest_neighbor_id": nn_id,
            "nearest_neighbor_name": nn_name,
            "nearest_neighbor_smiles": nn_smiles,
            "chembl_hits": chembl_hits,
            "pubchem_hits": pubchem_hits,
            "flag_threshold_used": self.flag_threshold,
            "fingerprint_method": _FINGERPRINT_METHOD,
            "query_fingerprint_hex": fp_hex,
            "apis_queried": apis_queried,
            "chembl_available": chembl_available,
            "pubchem_available": pubchem_available,
            "execution_time_ms": execution_time_ms,
            "error_reason": (
                "Both ChEMBL and PubChem APIs unavailable"
                if not chembl_available and not pubchem_available
                else None
            ),
        }

    # =========================================================================
    # SMILES extraction
    # =========================================================================

    def _extract_smiles(self, state: AgentState) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract the canonical SMILES from the ValidityTool result in state.

        Precedence:
            1. state.tool_results['ValidityTool']['smiles_canonical']
            2. raw_input if it's a non-empty string (fallback for tests)

        Returns:
            (smiles, None) on success
            (None, error_message) on failure
        """
        validity_result = state.tool_results.get("ValidityTool")

        if validity_result is not None:
            canonical = validity_result.get("smiles_canonical")
            if canonical and canonical.strip():
                return canonical.strip(), None
            # ValidityTool ran but produced no canonical SMILES (invalid molecule)
            return None, "ValidityTool result has no canonical SMILES (molecule may be invalid)"

        # No ValidityTool result — try raw_input as last resort
        raw = state.raw_input
        if isinstance(raw, str) and raw.strip():
            logger.debug(
                "SimilarityTool [%s]: no ValidityTool result, using raw_input as SMILES",
                state.molecule_id,
            )
            return raw.strip(), None

        return None, "No ValidityTool result and raw_input is not a SMILES string"

    # =========================================================================
    # ChEMBL API
    # =========================================================================

    def _query_chembl(
        self, smiles: str, molecule_id: str
    ) -> Tuple[List[dict], bool]:
        """
        Query the ChEMBL similarity REST API.

        Tries chembl_webresource_client first (handles encoding automatically),
        falls back to raw requests with explicit URL encoding.

        Threshold sent to ChEMBL is self.flag_threshold * 100 (integer percent).

        ChEMBL returns molecules in descending similarity order.
        We take the top max_results_per_api hits.

        Returns:
            (hits, available) where hits is a list of dicts and available
            indicates whether the API call succeeded.
        """
        if _CHEMBL_CLIENT_AVAILABLE:
            return self._query_chembl_via_client(smiles, molecule_id)
        return self._query_chembl_via_requests(smiles, molecule_id)

    def _query_chembl_via_client(
        self, smiles: str, molecule_id: str
    ) -> Tuple[List[dict], bool]:
        """Use chembl_webresource_client Python wrapper."""
        try:
            similarity = _chembl_client.similarity
            threshold_pct = int(self.flag_threshold * 100)

            # The client handles URL encoding of the SMILES internally
            results = similarity.filter(
                smiles=smiles,
                similarity=threshold_pct,
            ).only([
                "molecule_chembl_id",
                "pref_name",
                "similarity",
                "molecule_structures",
            ])

            hits = []
            for r in results[:self.max_results_per_api]:
                structures = r.get("molecule_structures") or {}
                canonical_smiles = structures.get("canonical_smiles", "")
                hits.append({
                    "source": "ChEMBL",
                    "id": r.get("molecule_chembl_id", ""),
                    "name": r.get("pref_name") or r.get("molecule_chembl_id", ""),
                    "tanimoto": float(r.get("similarity", 0.0)) / 100.0,
                    "smiles": canonical_smiles,
                })

            logger.debug(
                "SimilarityTool [%s]: ChEMBL (client) returned %d hits above %.0f%%",
                molecule_id, len(hits), self.flag_threshold * 100,
            )
            return hits, True

        except Exception as exc:
            logger.warning(
                "SimilarityTool [%s]: ChEMBL client query failed: %s", molecule_id, exc
            )
            # Fall back to raw requests
            return self._query_chembl_via_requests(smiles, molecule_id)

    def _query_chembl_via_requests(
        self, smiles: str, molecule_id: str
    ) -> Tuple[List[dict], bool]:
        """
        Fall-back: query ChEMBL REST API with raw requests + explicit URL encoding.

        SMILES special characters (#, %, /, @, +, [, ]) must be percent-encoded.
        urllib.parse.quote(smiles, safe='') handles this correctly.

        Endpoint: GET /similarity/{encoded_smiles}/{threshold_pct}
        """
        try:
            threshold_pct = int(self.flag_threshold * 100)
            # Explicit percent-encoding — critical for SMILES with /#@+[] etc.
            encoded_smiles = urllib.parse.quote(smiles, safe="")
            url = (
                f"{_CHEMBL_BASE_URL}/similarity/{encoded_smiles}/{threshold_pct}"
                f"?format=json&limit={self.max_results_per_api}"
            )

            resp = _requests.get(url, timeout=self.chembl_timeout)
            resp.raise_for_status()

            data = resp.json()
            molecules = data.get("molecules", [])

            hits = []
            for mol in molecules[:self.max_results_per_api]:
                structures = mol.get("molecule_structures") or {}
                canonical_smiles = structures.get("canonical_smiles", "")
                similarity_pct = mol.get("similarity", 0.0)
                hits.append({
                    "source": "ChEMBL",
                    "id": mol.get("molecule_chembl_id", ""),
                    "name": mol.get("pref_name") or mol.get("molecule_chembl_id", ""),
                    "tanimoto": float(similarity_pct) / 100.0,
                    "smiles": canonical_smiles,
                })

            logger.debug(
                "SimilarityTool [%s]: ChEMBL (requests) returned %d hits",
                molecule_id, len(hits),
            )
            return hits, True

        except _requests.exceptions.Timeout:
            logger.warning(
                "SimilarityTool [%s]: ChEMBL request timed out after %.1fs",
                molecule_id, self.chembl_timeout,
            )
            return [], False

        except _requests.exceptions.ConnectionError:
            logger.warning(
                "SimilarityTool [%s]: ChEMBL connection error (no network?)",
                molecule_id,
            )
            return [], False

        except Exception as exc:
            logger.warning(
                "SimilarityTool [%s]: ChEMBL query failed: %s", molecule_id, exc
            )
            return [], False

    # =========================================================================
    # PubChem API (async pattern)
    # =========================================================================

    def _query_pubchem(
        self, smiles: str, molecule_id: str
    ) -> Tuple[List[dict], bool]:
        """
        Query the PubChem PUG REST API for similar structures.

        PubChem similarity search is asynchronous:
            1. POST SMILES → receive listkey
            2. Poll GET /listkey/{key}/cids/JSON until status != running
            3. Fetch compound details for the returned CIDs

        Threshold: Threshold=85 (integer percent) sent as query param.

        Polling:
            - 1-second intervals
            - Maximum polls = pubchem_timeout / 1.0 (default 15 polls)

        Note on PubChem Tanimoto:
            PubChem uses their 881-bit fingerprint internally, not ECFP4.
            Returned similarity scores are proxies; both APIs are used
            conservatively (flag on either).

        Returns:
            (hits, available) where hits is a list of dicts and available
            indicates whether the API call succeeded.
        """
        try:
            threshold_pct = int(self.flag_threshold * 100)
            # Step 1: Submit similarity search
            encoded_smiles = urllib.parse.quote(smiles, safe="")
            submit_url = (
                f"{_PUBCHEM_BASE_URL}/compound/similarity/smiles"
                f"/{encoded_smiles}/JSON"
                f"?Threshold={threshold_pct}&MaxRecords={self.max_results_per_api}"
            )

            resp = _requests.post(
                submit_url,
                timeout=self.chembl_timeout,  # initial submission timeout
            )
            resp.raise_for_status()

            data = resp.json()
            waiting = data.get("Waiting")
            if waiting is None:
                # Occasionally PubChem returns results immediately
                return self._parse_pubchem_cids_to_hits(
                    data, molecule_id
                )

            listkey = waiting.get("ListKey")
            if not listkey:
                logger.warning(
                    "SimilarityTool [%s]: PubChem returned no ListKey", molecule_id
                )
                return [], False

            # Step 2: Poll for completion
            max_polls = int(self.pubchem_timeout)
            poll_url = (
                f"{_PUBCHEM_BASE_URL}/compound/listkey/{listkey}/cids/JSON"
            )

            for poll_num in range(max_polls):
                time.sleep(1.0)
                try:
                    poll_resp = _requests.get(
                        poll_url, timeout=self.chembl_timeout
                    )
                    poll_resp.raise_for_status()
                    poll_data = poll_resp.json()
                except Exception:
                    continue

                # Check if still waiting
                if "Waiting" in poll_data:
                    logger.debug(
                        "SimilarityTool [%s]: PubChem poll %d/%d — still waiting",
                        molecule_id, poll_num + 1, max_polls,
                    )
                    continue

                # Results ready
                cids = (
                    poll_data.get("IdentifierList", {}).get("CID", [])
                    or poll_data.get("InformationList", {}).get("Information", [])
                )
                hits = self._fetch_pubchem_compound_details(
                    cids[:self.max_results_per_api],
                    molecule_id=molecule_id,
                )
                logger.debug(
                    "SimilarityTool [%s]: PubChem returned %d hits after %d polls",
                    molecule_id, len(hits), poll_num + 1,
                )
                return hits, True

            # Timeout exhausted
            logger.warning(
                "SimilarityTool [%s]: PubChem polling timed out after %d polls (%.0fs)",
                molecule_id, max_polls, self.pubchem_timeout,
            )
            return [], False

        except _requests.exceptions.Timeout:
            logger.warning(
                "SimilarityTool [%s]: PubChem request timed out", molecule_id
            )
            return [], False

        except _requests.exceptions.ConnectionError:
            logger.warning(
                "SimilarityTool [%s]: PubChem connection error (no network?)",
                molecule_id,
            )
            return [], False

        except Exception as exc:
            logger.warning(
                "SimilarityTool [%s]: PubChem query failed: %s", molecule_id, exc
            )
            return [], False

    def _parse_pubchem_cids_to_hits(
        self, data: dict, molecule_id: str
    ) -> Tuple[List[dict], bool]:
        """Handle case where PubChem returns results immediately (no polling)."""
        try:
            cids = (
                data.get("IdentifierList", {}).get("CID", [])
                or data.get("InformationList", {}).get("Information", [])
            )
            if not cids:
                return [], True  # available but no hits
            hits = self._fetch_pubchem_compound_details(
                cids[:self.max_results_per_api], molecule_id=molecule_id
            )
            return hits, True
        except Exception as exc:
            logger.warning(
                "SimilarityTool [%s]: PubChem immediate-result parsing failed: %s",
                molecule_id, exc,
            )
            return [], False

    def _fetch_pubchem_compound_details(
        self, cids: List[int], molecule_id: str
    ) -> List[dict]:
        """
        Fetch compound names and SMILES for a list of PubChem CIDs.

        PubChem doesn't return Tanimoto scores in its CID list response.
        We set tanimoto=1.0 as a conservative placeholder for any CID that
        appeared in the above-threshold results. This ensures the FLAG logic
        behaves correctly: if any hit appeared above the threshold, it counts.

        Args:
            cids: List of integer CIDs (already trimmed to max_results)
            molecule_id: For logging

        Returns:
            List of hit dicts with source='PubChem'
        """
        if not cids:
            return []

        hits = []
        try:
            cid_str = ",".join(str(c) for c in cids)
            props_url = (
                f"{_PUBCHEM_BASE_URL}/compound/cid/{cid_str}/property"
                f"/IUPACName,IsomericSMILES/JSON"
            )
            resp = _requests.get(props_url, timeout=self.chembl_timeout)
            resp.raise_for_status()

            props_data = resp.json()
            properties = props_data.get("PropertyTable", {}).get("Properties", [])

            for prop in properties:
                cid = prop.get("CID", "")
                name = prop.get("IUPACName", f"CID{cid}")
                smiles = prop.get("IsomericSMILES", "")
                # PubChem's API returns compounds above our threshold;
                # tanimoto=self.flag_threshold is the minimum guaranteed.
                # We conservatively report it as the threshold itself.
                hits.append({
                    "source": "PubChem",
                    "id": str(cid),
                    "name": name,
                    "tanimoto": self.flag_threshold,  # minimum — see docstring
                    "smiles": smiles,
                })

        except Exception as exc:
            logger.warning(
                "SimilarityTool [%s]: fetching PubChem compound details failed: %s",
                molecule_id, exc,
            )

        return hits

    # =========================================================================
    # Nearest-neighbor selection
    # =========================================================================

    def _find_nearest_neighbor(
        self,
        chembl_hits: List[dict],
        pubchem_hits: List[dict],
    ) -> Tuple[float, Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Find the globally highest-Tanimoto hit across both API results.

        Returns:
            Tuple of (tanimoto, source, id, name, smiles) for the best hit,
            or (0.0, None, None, None, None) if no hits from either API.
        """
        all_hits = chembl_hits + pubchem_hits
        if not all_hits:
            return 0.0, None, None, None, None

        best = max(all_hits, key=lambda h: h.get("tanimoto", 0.0))
        return (
            best.get("tanimoto", 0.0),
            best.get("source"),
            best.get("id"),
            best.get("name"),
            best.get("smiles"),
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _elapsed_ms(self, start_time: float) -> float:
        """Return elapsed time in milliseconds since start_time (monotonic)."""
        return round((time.monotonic() - start_time) * 1000, 2)

    def _error_result(
        self,
        molecule_id: str,
        query_smiles: Optional[str],
        error_reason: str,
        execution_time_ms: float,
    ) -> Dict[str, Any]:
        """Build a standardized ERROR result dict."""
        return {
            "tool_name": self.name,
            "molecule_id": molecule_id,
            "query_smiles": query_smiles,
            "similarity_decision": "ERROR",
            "nearest_neighbor_tanimoto": 0.0,
            "nearest_neighbor_source": None,
            "nearest_neighbor_id": None,
            "nearest_neighbor_name": None,
            "nearest_neighbor_smiles": None,
            "chembl_hits": [],
            "pubchem_hits": [],
            "flag_threshold_used": self.flag_threshold,
            "fingerprint_method": _FINGERPRINT_METHOD,
            "query_fingerprint_hex": None,
            "apis_queried": [],
            "chembl_available": False,
            "pubchem_available": False,
            "execution_time_ms": execution_time_ms,
            "error_reason": error_reason,
        }