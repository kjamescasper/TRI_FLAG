"""
tools/similarity_tool.py

Similarity-based IP-risk screening via ChEMBL, SureChEMBL, and PubChem.

Queries three public cheminformatics databases at runtime:

  - ChEMBL REST API (~2.4M compounds): approved and bioactive drug-like
    chemistry. A hit here means structural similarity to a known drug or
    clinical candidate — the strongest IP concern.

  - SureChEMBL (~28M compounds): structures extracted directly from patent
    literature by the EBI. A hit here means the scaffold appears in a filed
    patent. This is the scientifically correct source for IP-risk screening
    in a drug discovery context.

  - PubChem PUG REST API (~115M compounds): broadest structure coverage
    including research compounds, industrial chemicals, and literature
    molecules. PubChem hits are stored for reference but do NOT trigger
    FLAG decisions — the database is too broad to be meaningful as an IP
    signal. Most novel drug-like scaffolds have some PubChem neighbour.

FLAG decision logic (in priority order):
    ChEMBL Tanimoto  >= flag_threshold  →  FLAG  (drug/clinical candidate)
    SureChEMBL Tanimoto >= flag_threshold →  FLAG  (patented compound)
    PubChem Tanimoto >= any threshold    →  stored in pubchem_hits, no FLAG

Design principles:
  - FLAG-only (never DISCARD) — similarity is a proxy, not a legal judgment.
    A chemist or patent attorney must make the final IP determination.
  - FLAG-and-continue — consistent with SAScoreTool behaviour.
  - All API failures are non-terminal. If both flagging APIs fail, ERROR is
    returned and the PolicyEngine flags conservatively.
  - SimilarityTool always runs, even if SAScoreTool flagged the molecule.
    Rationale: synthetically challenging + novel differs from
    synthetically challenging + known drug.
  - SMILES are read from state.tool_results['ValidityTool']['smiles_canonical']
    to ensure consistency with the rest of the pipeline.

Performance note:
  Live API calls add ~3–15s per molecule depending on network and server
  load. Use --no-similarity for fast offline development runs.

SureChEMBL API note (updated Week 11):
  The old /api/search/similarity endpoint was removed in the 2024 rebuild.
  The new API is async: POST /search/structure → poll /search/{hash}/status
  → GET /search/{hash}/results. The similarity threshold is applied
  client-side after fetching results.

Literature:
  Maggiora G et al. (2014). J. Med. Chem. 57(8), 3186-3204.
    DOI: 10.1021/jm401411z — ECFP4 Tanimoto is standard for IP-risk.
  Bender A, Glen RC (2004). Org. Biomol. Chem. 2, 3204-3218.
    DOI: 10.1039/b409813g — 0.85 threshold minimizes false positives.

Week: 9 (SureChEMBL added; PubChem demoted to informational)
Week: 11 (SureChEMBL updated to new 2024 async API)
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
# requests import guard
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
# Optional chembl_webresource_client — preferred for ChEMBL queries
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
_CHEMBL_BASE_URL     = "https://www.ebi.ac.uk/chembl/api/data"
_SURECHEMBL_BASE_URL = "https://www.surechembl.org/api"   # updated: new 2024 API base
_PUBCHEM_BASE_URL    = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

_FINGERPRINT_METHOD = "Morgan_ECFP4_r2_2048bits"


class SimilarityTool(Tool):
    """
    Chemical similarity screening tool using live public cheminformatics APIs.

    FLAG sources (drive the decision):
        ChEMBL    — approved drugs and bioactive compounds (~2.4M)
        SureChEMBL — compounds extracted from patent literature (~28M)

    Informational source (stored, never flags):
        PubChem   — all publicly known structures (~115M)

    A Tanimoto >= flag_threshold against ChEMBL or SureChEMBL triggers a
    FLAG annotation on AgentState. PubChem results are stored in
    pubchem_hits for reference only.

    Result dict stored in state.tool_results['SimilarityTool']:
        {
            'tool_name':                  str,
            'molecule_id':                str,
            'query_smiles':               str,
            'similarity_decision':        'PASS' | 'FLAG' | 'ERROR',
            'flag_source':                'ChEMBL' | 'SureChEMBL' | None,
            'nearest_neighbor_tanimoto':  float,
            'nearest_neighbor_source':    'ChEMBL' | 'SureChEMBL' | None,
            'nearest_neighbor_id':        str | None,
            'nearest_neighbor_name':      str | None,
            'nearest_neighbor_smiles':    str | None,
            'chembl_hits':                List[dict],
            'surechembl_hits':            List[dict],
            'pubchem_hits':               List[dict],   # informational only
            'flag_threshold_used':        float,
            'fingerprint_method':         str,
            'query_fingerprint_hex':      str | None,
            'apis_queried':               List[str],
            'chembl_available':           bool,
            'surechembl_available':       bool,
            'pubchem_available':          bool,
            'execution_time_ms':          float,
            'error_reason':               str | None,
        }
    """

    name = "SimilarityTool"

    def __init__(
        self,
        flag_threshold: float = 0.90,
        use_chembl: bool = True,
        use_surechembl: bool = True,
        use_pubchem: bool = True,
        chembl_timeout: float = 10.0,
        surechembl_timeout: float = 15.0,   # increased: new API needs poll time
        pubchem_timeout: float = 15.0,
        max_results_per_api: int = 5,
    ):
        """
        Initialise SimilarityTool.

        Args:
            flag_threshold:    Tanimoto >= this against ChEMBL or SureChEMBL
                               triggers FLAG. Default 0.90.
            use_chembl:        Query ChEMBL (flagging source).
            use_surechembl:    Query SureChEMBL (flagging source).
            use_pubchem:       Query PubChem (informational only — never flags).
            chembl_timeout:    Seconds before ChEMBL request times out.
            surechembl_timeout: Max seconds for SureChEMBL async flow
                               (submit + poll + fetch). Default 15.
            pubchem_timeout:   Seconds total for PubChem async polling.
            max_results_per_api: Maximum hits to retrieve from each API.
        """
        self.flag_threshold       = flag_threshold
        self.use_chembl           = use_chembl
        self.use_surechembl       = use_surechembl
        self.use_pubchem          = use_pubchem
        self.chembl_timeout       = chembl_timeout
        self.surechembl_timeout   = surechembl_timeout
        self.pubchem_timeout      = pubchem_timeout
        self.max_results_per_api  = max_results_per_api
        self.description = (
            "Similarity-based IP-risk screening via ChEMBL and SureChEMBL "
            f"(flag threshold: Tanimoto >= {flag_threshold:.2f}). "
            "PubChem searched for reference only."
        )
        logger.info(
            "SimilarityTool initialized: flag_threshold=%.2f, "
            "use_chembl=%s, use_surechembl=%s, use_pubchem=%s, max_results=%d",
            flag_threshold, use_chembl, use_surechembl, use_pubchem,
            max_results_per_api,
        )

    # =========================================================================
    # Public entry point
    # =========================================================================

    def run(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute similarity screening for the molecule in state.

        Reads canonical SMILES from state.tool_results['ValidityTool'].
        Queries ChEMBL and SureChEMBL for FLAG decisions; PubChem for
        reference. Always runs regardless of whether prior tools flagged.

        Args:
            state: AgentState with ValidityTool result already stored.

        Returns:
            Result dictionary (see class docstring for full schema).
        """
        start_time  = time.monotonic()
        molecule_id = state.molecule_id

        logger.info("SimilarityTool: starting for molecule_id=%s", molecule_id)

        # Step 1: Extract canonical SMILES
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

        # Step 2: Local fingerprint for provenance
        local_fp = morgan_fingerprint_from_smiles(smiles)
        fp_hex   = fingerprint_to_hex(local_fp)

        # Step 3: Query APIs
        chembl_hits:     List[dict] = []
        surechembl_hits: List[dict] = []
        pubchem_hits:    List[dict] = []
        chembl_available     = False
        surechembl_available = False
        pubchem_available    = False
        apis_queried: List[str] = []

        if self.use_chembl and _REQUESTS_AVAILABLE:
            chembl_hits, chembl_available = self._query_chembl(smiles, molecule_id)
            if chembl_available:
                apis_queried.append("ChEMBL")

        if self.use_surechembl and _REQUESTS_AVAILABLE:
            surechembl_hits, surechembl_available = self._query_surechembl(
                smiles, molecule_id
            )
            if surechembl_available:
                apis_queried.append("SureChEMBL")

        if self.use_pubchem and _REQUESTS_AVAILABLE:
            pubchem_hits, pubchem_available = self._query_pubchem(smiles, molecule_id)
            if pubchem_available:
                apis_queried.append("PubChem")

        # Step 4: FLAG decision — ChEMBL and SureChEMBL only
        flagging_hits = chembl_hits + surechembl_hits
        nn_tanimoto, nn_source, nn_id, nn_name, nn_smiles = (
            self._find_nearest_neighbor(flagging_hits)
        )

        both_flagging_apis_failed = (
            (self.use_chembl     and not chembl_available) and
            (self.use_surechembl and not surechembl_available)
        )

        if not _REQUESTS_AVAILABLE or both_flagging_apis_failed:
            similarity_decision = "ERROR"
            flag_source = None
        elif nn_tanimoto >= self.flag_threshold:
            similarity_decision = "FLAG"
            flag_source = nn_source
        else:
            similarity_decision = "PASS"
            flag_source = None

        execution_time_ms = self._elapsed_ms(start_time)

        logger.info(
            "SimilarityTool [%s]: decision=%s, nearest_neighbor=%.3f from %s",
            molecule_id, similarity_decision, nn_tanimoto, nn_source or "N/A",
        )

        return {
            "tool_name":                 self.name,
            "molecule_id":               molecule_id,
            "query_smiles":              smiles,
            "similarity_decision":       similarity_decision,
            "flag_source":               flag_source,
            "nearest_neighbor_tanimoto": nn_tanimoto,
            "nearest_neighbor_source":   nn_source,
            "nearest_neighbor_id":       nn_id,
            "nearest_neighbor_name":     nn_name,
            "nearest_neighbor_smiles":   nn_smiles,
            "chembl_hits":               chembl_hits,
            "surechembl_hits":           surechembl_hits,
            "pubchem_hits":              pubchem_hits,
            "flag_threshold_used":       self.flag_threshold,
            "fingerprint_method":        _FINGERPRINT_METHOD,
            "query_fingerprint_hex":     fp_hex,
            "apis_queried":              apis_queried,
            "chembl_available":          chembl_available,
            "surechembl_available":      surechembl_available,
            "pubchem_available":         pubchem_available,
            "execution_time_ms":         execution_time_ms,
            "error_reason": (
                "Both ChEMBL and SureChEMBL APIs unavailable"
                if both_flagging_apis_failed else None
            ),
        }

    # =========================================================================
    # SMILES extraction
    # =========================================================================

    def _extract_smiles(
        self, state: AgentState
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract canonical SMILES from the ValidityTool result in state.

        Precedence:
            1. state.tool_results['ValidityTool']['smiles_canonical']
            2. raw_input if it is a non-empty string (fallback for tests)

        Returns:
            (smiles, None) on success
            (None, error_message) on failure
        """
        validity_result = state.tool_results.get("ValidityTool")

        if validity_result is not None:
            canonical = validity_result.get("smiles_canonical")
            if canonical and canonical.strip():
                return canonical.strip(), None
            return (
                None,
                "ValidityTool result has no canonical SMILES (molecule may be invalid)",
            )

        raw = state.raw_input
        if isinstance(raw, str) and raw.strip():
            logger.debug(
                "SimilarityTool [%s]: no ValidityTool result, using raw_input",
                state.molecule_id,
            )
            return raw.strip(), None

        return None, "No ValidityTool result and raw_input is not a SMILES string"

    # =========================================================================
    # ChEMBL API  (flagging source)
    # =========================================================================

    def _query_chembl(
        self, smiles: str, molecule_id: str
    ) -> Tuple[List[dict], bool]:
        """
        Query the ChEMBL similarity REST API.

        Tries chembl_webresource_client first; falls back to raw requests.
        Threshold sent to ChEMBL is self.flag_threshold * 100 (integer %).
        """
        if _CHEMBL_CLIENT_AVAILABLE:
            return self._query_chembl_via_client(smiles, molecule_id)
        return self._query_chembl_via_requests(smiles, molecule_id)

    def _query_chembl_via_client(
        self, smiles: str, molecule_id: str
    ) -> Tuple[List[dict], bool]:
        """Use chembl_webresource_client Python wrapper."""
        try:
            similarity    = _chembl_client.similarity
            threshold_pct = int(self.flag_threshold * 100)

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
                hits.append({
                    "source":   "ChEMBL",
                    "id":       r.get("molecule_chembl_id", ""),
                    "name":     r.get("pref_name") or r.get("molecule_chembl_id", ""),
                    "tanimoto": float(r.get("similarity", 0.0)) / 100.0,
                    "smiles":   structures.get("canonical_smiles", ""),
                })

            logger.debug(
                "SimilarityTool [%s]: ChEMBL (client) returned %d hits above %.0f%%",
                molecule_id, len(hits), self.flag_threshold * 100,
            )
            return hits, True

        except Exception as exc:
            logger.warning(
                "SimilarityTool [%s]: ChEMBL client query failed: %s — "
                "retrying with raw requests",
                molecule_id, exc,
            )
            return self._query_chembl_via_requests(smiles, molecule_id)

    def _query_chembl_via_requests(
        self, smiles: str, molecule_id: str
    ) -> Tuple[List[dict], bool]:
        """
        Fallback: query ChEMBL REST API with raw requests.

        SMILES special characters must be percent-encoded.
        urllib.parse.quote(smiles, safe='') handles this correctly.
        """
        try:
            threshold_pct  = int(self.flag_threshold * 100)
            encoded_smiles = urllib.parse.quote(smiles, safe="")
            url = (
                f"{_CHEMBL_BASE_URL}/similarity/{encoded_smiles}/{threshold_pct}"
                f"?format=json&limit={self.max_results_per_api}"
            )

            resp = _requests.get(url, timeout=self.chembl_timeout)
            resp.raise_for_status()

            hits = []
            for mol in resp.json().get("molecules", [])[:self.max_results_per_api]:
                structures = mol.get("molecule_structures") or {}
                hits.append({
                    "source":   "ChEMBL",
                    "id":       mol.get("molecule_chembl_id", ""),
                    "name":     mol.get("pref_name") or mol.get("molecule_chembl_id", ""),
                    "tanimoto": float(mol.get("similarity", 0.0)) / 100.0,
                    "smiles":   structures.get("canonical_smiles", ""),
                })

            logger.debug(
                "SimilarityTool [%s]: ChEMBL (requests) returned %d hits",
                molecule_id, len(hits),
            )
            return hits, True

        except _requests.exceptions.Timeout:
            logger.warning(
                "SimilarityTool [%s]: ChEMBL timed out after %.1fs",
                molecule_id, self.chembl_timeout,
            )
            return [], False
        except _requests.exceptions.ConnectionError:
            logger.warning(
                "SimilarityTool [%s]: ChEMBL connection error (offline?)",
                molecule_id,
            )
            return [], False
        except Exception as exc:
            logger.warning(
                "SimilarityTool [%s]: ChEMBL query failed: %s", molecule_id, exc
            )
            return [], False

    # =========================================================================
    # SureChEMBL API  (flagging source — patent literature)
    # Updated Week 11: new 2024 async API replaces old /search/similarity GET
    # =========================================================================

    def _query_surechembl(
        self, smiles: str, molecule_id: str
    ) -> Tuple[List[dict], bool]:
        """
        Query the SureChEMBL similarity API (new 2024 async API).

        The old synchronous GET /api/search/similarity endpoint was removed
        in the 2024 SureChEMBL rebuild. The new flow is:

            1. POST /search/structure
               Body: {"StructureSearchRequest": {"struct": smiles,
                      "structSearchType": "similarity", "maxResults": N}}
               Response: {"data": {"hash": "<uuid>"}}

            2. Poll GET /search/{hash}/status
               until data.message contains "finished"

            3. GET /search/{hash}/results?page=0&max_results=N
               Response: data.results.structures (list of compound dicts)
               Each dict has "similarity" (string float in [0,1])

        Threshold filtering is applied client-side — the server returns all
        similarity results; we keep only hits >= flag_threshold.

        Args:
            smiles:      Canonical SMILES string.
            molecule_id: For logging.

        Returns:
            (hits, available)
        """
        try:
            # ── Step 1: submit search ───────────────────────────────────────
            submit_resp = _requests.post(
                f"{_SURECHEMBL_BASE_URL}/search/structure",
                json={
                    "StructureSearchRequest": {
                        "struct":           smiles,
                        "structSearchType": "similarity",
                        # Request more than needed so client-side threshold
                        # filtering has enough candidates to work with
                        "maxResults":       self.max_results_per_api * 20,
                    }
                },
                timeout=self.surechembl_timeout,
            )
            submit_resp.raise_for_status()
            search_hash = submit_resp.json().get("data", {}).get("hash")

            if not search_hash:
                logger.warning(
                    "SimilarityTool [%s]: SureChEMBL returned no hash",
                    molecule_id,
                )
                return [], False

            # ── Step 2: poll until finished ─────────────────────────────────
            status_url = f"{_SURECHEMBL_BASE_URL}/search/{search_hash}/status"
            max_polls  = int(self.surechembl_timeout)  # 1 poll per second
            finished   = False

            for _ in range(max_polls):
                time.sleep(1.0)
                try:
                    st = _requests.get(status_url, timeout=self.surechembl_timeout)
                    st.raise_for_status()
                    msg = st.json().get("data", {}).get("message", "")
                    if "finished" in msg.lower():
                        finished = True
                        break
                except Exception:
                    continue

            if not finished:
                logger.warning(
                    "SimilarityTool [%s]: SureChEMBL search did not finish "
                    "within %d seconds",
                    molecule_id, max_polls,
                )
                return [], False

            # ── Step 3: fetch results ───────────────────────────────────────
            results_resp = _requests.get(
                f"{_SURECHEMBL_BASE_URL}/search/{search_hash}/results",
                params={"page": 0, "max_results": self.max_results_per_api * 20},
                timeout=self.surechembl_timeout,
            )
            results_resp.raise_for_status()
            structures = (
                results_resp.json()
                .get("data", {})
                .get("results", {})
                .get("structures", [])
            )

            # ── Step 4: client-side threshold filter ────────────────────────
            hits = []
            for s in structures:
                try:
                    tanimoto = float(s.get("similarity", 0.0))
                except (TypeError, ValueError):
                    continue

                if tanimoto < self.flag_threshold:
                    continue

                hits.append({
                    "source":   "SureChEMBL",
                    "id":       str(s.get("chemical_id", s.get("id", ""))),
                    "name":     s.get("name") or str(s.get("chemical_id", "")),
                    "tanimoto": tanimoto,
                    "smiles":   s.get("smiles", ""),
                })

                if len(hits) >= self.max_results_per_api:
                    break

            logger.debug(
                "SimilarityTool [%s]: SureChEMBL returned %d hits above %.2f",
                molecule_id, len(hits), self.flag_threshold,
            )
            return hits, True

        except _requests.exceptions.Timeout:
            logger.warning(
                "SimilarityTool [%s]: SureChEMBL timed out after %.1fs",
                molecule_id, self.surechembl_timeout,
            )
            return [], False
        except _requests.exceptions.ConnectionError:
            logger.warning(
                "SimilarityTool [%s]: SureChEMBL connection error (offline?)",
                molecule_id,
            )
            return [], False
        except Exception as exc:
            logger.warning(
                "SimilarityTool [%s]: SureChEMBL query failed: %s",
                molecule_id, exc,
            )
            return [], False

    # =========================================================================
    # PubChem API  (informational only — does not drive FLAG decisions)
    # =========================================================================

    def _query_pubchem(
        self, smiles: str, molecule_id: str
    ) -> Tuple[List[dict], bool]:
        """
        Query the PubChem PUG REST API for similar structures.

        Results are stored in pubchem_hits for reference and rationale
        context. They are NEVER used in the FLAG decision — see class
        docstring for rationale.

        PubChem similarity search is asynchronous:
            1. POST SMILES → receive ListKey
            2. Poll GET /listkey/{key}/cids/JSON until status != running
            3. Fetch compound details for the returned CIDs

        Threshold sent: Threshold=85 (integer %) regardless of flag_threshold,
        since PubChem hits are informational and a lower threshold gives more
        useful context.
        """
        try:
            # Fixed informational threshold — independent of flag_threshold
            _PUBCHEM_INFO_THRESHOLD = 85

            encoded_smiles = urllib.parse.quote(smiles, safe="")
            submit_url = (
                f"{_PUBCHEM_BASE_URL}/compound/similarity/smiles"
                f"/{encoded_smiles}/JSON"
                f"?Threshold={_PUBCHEM_INFO_THRESHOLD}"
                f"&MaxRecords={self.max_results_per_api}"
            )

            resp = _requests.post(submit_url, timeout=self.chembl_timeout)
            resp.raise_for_status()

            data = resp.json()
            waiting = data.get("Waiting")
            if waiting is None:
                return self._parse_pubchem_immediate(data, molecule_id)

            listkey = waiting.get("ListKey")
            if not listkey:
                logger.warning(
                    "SimilarityTool [%s]: PubChem returned no ListKey", molecule_id
                )
                return [], False

            poll_url  = f"{_PUBCHEM_BASE_URL}/compound/listkey/{listkey}/cids/JSON"
            max_polls = int(self.pubchem_timeout)

            for poll_num in range(max_polls):
                time.sleep(1.0)
                try:
                    poll_resp = _requests.get(poll_url, timeout=self.chembl_timeout)
                    poll_resp.raise_for_status()
                    poll_data = poll_resp.json()
                except Exception:
                    continue

                if "Waiting" in poll_data:
                    logger.debug(
                        "SimilarityTool [%s]: PubChem poll %d/%d — waiting",
                        molecule_id, poll_num + 1, max_polls,
                    )
                    continue

                cids = (
                    poll_data.get("IdentifierList", {}).get("CID", [])
                    or poll_data.get("InformationList", {}).get("Information", [])
                )
                hits = self._fetch_pubchem_details(
                    cids[:self.max_results_per_api], molecule_id
                )
                logger.debug(
                    "SimilarityTool [%s]: PubChem (informational) returned %d hits",
                    molecule_id, len(hits),
                )
                return hits, True

            logger.warning(
                "SimilarityTool [%s]: PubChem polling timed out after %d polls",
                molecule_id, max_polls,
            )
            return [], False

        except _requests.exceptions.Timeout:
            logger.warning("SimilarityTool [%s]: PubChem timed out", molecule_id)
            return [], False
        except _requests.exceptions.ConnectionError:
            logger.warning(
                "SimilarityTool [%s]: PubChem connection error (offline?)",
                molecule_id,
            )
            return [], False
        except Exception as exc:
            logger.warning(
                "SimilarityTool [%s]: PubChem query failed: %s", molecule_id, exc
            )
            return [], False

    def _parse_pubchem_immediate(
        self, data: dict, molecule_id: str
    ) -> Tuple[List[dict], bool]:
        """Handle the case where PubChem returns results without polling."""
        try:
            cids = (
                data.get("IdentifierList", {}).get("CID", [])
                or data.get("InformationList", {}).get("Information", [])
            )
            if not cids:
                return [], True
            hits = self._fetch_pubchem_details(
                cids[:self.max_results_per_api], molecule_id
            )
            return hits, True
        except Exception as exc:
            logger.warning(
                "SimilarityTool [%s]: PubChem immediate-result parsing failed: %s",
                molecule_id, exc,
            )
            return [], False

    def _fetch_pubchem_details(
        self, cids: List[int], molecule_id: str
    ) -> List[dict]:
        """
        Fetch IUPAC names and SMILES for a list of PubChem CIDs.

        PubChem does not return Tanimoto scores in its CID list response.
        We do not set tanimoto=flag_threshold here because PubChem hits are
        informational only and are never compared against the flag threshold.
        Tanimoto is stored as 0.0 with a note to avoid misleading consumers
        of the result dict.
        """
        if not cids:
            return []

        hits = []
        try:
            cid_str   = ",".join(str(c) for c in cids)
            props_url = (
                f"{_PUBCHEM_BASE_URL}/compound/cid/{cid_str}/property"
                f"/IUPACName,IsomericSMILES/JSON"
            )
            resp = _requests.get(props_url, timeout=self.chembl_timeout)
            resp.raise_for_status()

            for prop in resp.json().get("PropertyTable", {}).get("Properties", []):
                cid = prop.get("CID", "")
                hits.append({
                    "source":          "PubChem",
                    "id":              str(cid),
                    "name":            prop.get("IUPACName", f"CID{cid}"),
                    "tanimoto":        0.0,   # PubChem does not return Tanimoto
                    "smiles":          prop.get("IsomericSMILES", ""),
                    "informational":   True,  # explicit marker — never used for FLAG
                })

        except Exception as exc:
            logger.warning(
                "SimilarityTool [%s]: fetching PubChem details failed: %s",
                molecule_id, exc,
            )

        return hits

    # =========================================================================
    # Nearest-neighbour selection  (flagging sources only)
    # =========================================================================

    def _find_nearest_neighbor(
        self,
        flagging_hits: List[dict],
    ) -> Tuple[float, Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Find the highest-Tanimoto hit across ChEMBL and SureChEMBL results.

        PubChem hits are deliberately excluded — they are informational only
        and must never influence the FLAG decision.

        Args:
            flagging_hits: Combined ChEMBL + SureChEMBL hits.

        Returns:
            (tanimoto, source, id, name, smiles) for the best hit,
            or (0.0, None, None, None, None) if no hits.
        """
        if not flagging_hits:
            return 0.0, None, None, None, None

        best = max(flagging_hits, key=lambda h: h.get("tanimoto", 0.0))
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
        """Return elapsed wall-clock time in milliseconds."""
        return round((time.monotonic() - start_time) * 1000, 2)

    def _error_result(
        self,
        molecule_id: str,
        query_smiles: Optional[str],
        error_reason: str,
        execution_time_ms: float,
    ) -> Dict[str, Any]:
        """Build a standardised ERROR result dict."""
        return {
            "tool_name":                 self.name,
            "molecule_id":               molecule_id,
            "query_smiles":              query_smiles,
            "similarity_decision":       "ERROR",
            "flag_source":               None,
            "nearest_neighbor_tanimoto": 0.0,
            "nearest_neighbor_source":   None,
            "nearest_neighbor_id":       None,
            "nearest_neighbor_name":     None,
            "nearest_neighbor_smiles":   None,
            "chembl_hits":               [],
            "surechembl_hits":           [],
            "pubchem_hits":              [],
            "flag_threshold_used":       self.flag_threshold,
            "fingerprint_method":        _FINGERPRINT_METHOD,
            "query_fingerprint_hex":     None,
            "apis_queried":              [],
            "chembl_available":          False,
            "surechembl_available":      False,
            "pubchem_available":         False,
            "execution_time_ms":         execution_time_ms,
            "error_reason":              error_reason,
        }