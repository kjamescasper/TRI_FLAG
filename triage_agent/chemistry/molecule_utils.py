"""
chemistry/molecule_utils.py

Shared SMILES canonicalization and molecule utility helpers.

Used by fingerprints.py (and other tools) to ensure consistent SMILES
representation before any downstream computation. Thin wrapper — no domain
logic, no policy decisions.

Design:
    - All functions are pure (no side effects)
    - Failures return None / (None, error_str) rather than raising
    - RDKit dependency is guarded with a clear error message

Week: 5
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RDKit import guard — consistent with ValidityTool
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    logger.warning(
        "RDKit not available — molecule_utils functions will return None. "
        "Install with: conda install -c conda-forge rdkit"
    )


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Return the canonical SMILES for a given SMILES string, or None on failure.

    NOTE: SimilarityTool reads canonical SMILES from
    state.tool_results['ValidityTool']['smiles_canonical'] rather than
    calling this function directly. This utility exists for use cases where
    an upstream validity result is not available (e.g. standalone fingerprint
    computation in tests).

    Args:
        smiles: Input SMILES string (canonical or non-canonical)

    Returns:
        Canonical SMILES string if valid, None if RDKit cannot parse it
        or if SMILES is empty/None.

    Examples:
        >>> canonicalize_smiles("OCC")    # non-canonical ethanol
        "CCO"
        >>> canonicalize_smiles("c1ccccc1")
        "c1ccccc1"
        >>> canonicalize_smiles("INVALID")
        None
    """
    if not _RDKIT_AVAILABLE:
        logger.error("canonicalize_smiles: RDKit not available")
        return None

    if not smiles or not smiles.strip():
        return None

    try:
        mol = Chem.MolFromSmiles(smiles.strip(), sanitize=True)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as exc:
        logger.debug("canonicalize_smiles failed for %r: %s", smiles, exc)
        return None


def smiles_to_mol(smiles: str):
    """
    Parse a SMILES string into an RDKit Mol object.

    Args:
        smiles: SMILES string to parse

    Returns:
        RDKit Mol object if valid, None otherwise

    Notes:
        Callers should check for None before using the returned object.
        Sanitization is always performed (catches valence errors, etc.).
    """
    if not _RDKIT_AVAILABLE:
        return None

    if not smiles or not smiles.strip():
        return None

    try:
        mol = Chem.MolFromSmiles(smiles.strip(), sanitize=True)
        return mol
    except Exception as exc:
        logger.debug("smiles_to_mol failed for %r: %s", smiles, exc)
        return None


def validate_and_canonicalize(smiles: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Validate a SMILES string and return its canonical form.

    Returns a (canonical_smiles, error_message) tuple — consistent with the
    return style of validate_smiles() in validity_tool.py.

    Args:
        smiles: SMILES string to validate and canonicalize

    Returns:
        Tuple of:
            - canonical_smiles (str): Canonical SMILES if valid, else None
            - error_message (str): Error description if invalid, else None

    Examples:
        >>> validate_and_canonicalize("OCC")
        ("CCO", None)
        >>> validate_and_canonicalize("")
        (None, "SMILES string is empty")
        >>> validate_and_canonicalize("C(((")
        (None, "RDKit failed to parse SMILES: 'C((('")
    """
    if not _RDKIT_AVAILABLE:
        return None, "RDKit not available"

    if not smiles or not smiles.strip():
        return None, "SMILES string is empty"

    try:
        mol = Chem.MolFromSmiles(smiles.strip(), sanitize=True)
        if mol is None:
            return None, f"RDKit failed to parse SMILES: '{smiles}'"

        if mol.GetNumAtoms() == 0:
            return None, "Molecule has zero atoms (empty structure)"

        canonical = Chem.MolToSmiles(mol, canonical=True)
        return canonical, None

    except Exception as exc:
        return None, f"RDKit exception: {exc}"