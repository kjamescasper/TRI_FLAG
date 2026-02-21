"""
chemistry/fingerprints.py

Morgan ECFP4 fingerprint computation and Tanimoto similarity utilities.

Implements Morgan circular fingerprints (ECFP4 variant) for molecular
similarity estimation. Used by SimilarityTool to:
  1. Validate that canonical SMILES is computationally tractable before
     making API calls.
  2. Store the local fingerprint hex in AgentState for provenance tracking,
     enabling reproducibility audits without re-querying external databases.

Note on fingerprint concordance:
    The ChEMBL REST API uses RDKit Morgan fingerprints internally, making
    our local computation directly comparable. PubChem uses their own 881-bit
    fingerprint; Tanimoto scores from PubChem are proxies, not exact ECFP4
    values. Both are used conservatively (flag on either).

Literature:
    Rogers D, Hahn M (2010). Extended-Connectivity Fingerprints.
    J. Chem. Inf. Model. 50(5), 742-754. DOI: 10.1021/ci100050t
    — Canonical ECFP reference; ECFP4 (radius=2) is the standard for
      drug-like molecular similarity.

    Maggiora G et al. (2014). Molecular Similarity in Medicinal Chemistry.
    J. Med. Chem. 57(8), 3186-3204. DOI: 10.1021/jm401411z
    — Tanimoto + ECFP4 is the standard for IP-risk similarity assessment.

Week: 5
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RDKit import guard
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    logger.warning(
        "RDKit not available — fingerprints.py functions will return None. "
        "Install with: conda install -c conda-forge rdkit"
    )

# Default fingerprint parameters (ECFP4)
_DEFAULT_RADIUS = 2       # radius=2 → ECFP4
_DEFAULT_N_BITS = 2048    # 2048-bit vector (standard for drug-like molecules)


# ---------------------------------------------------------------------------
# Core fingerprint computation
# ---------------------------------------------------------------------------

def morgan_fingerprint(mol, radius: int = _DEFAULT_RADIUS, n_bits: int = _DEFAULT_N_BITS):
    """
    Compute a Morgan (ECFP4) fingerprint for an RDKit Mol object.

    Args:
        mol: RDKit Mol object (already sanitized)
        radius: Morgan radius. Default 2 → ECFP4.
        n_bits: Fingerprint bit vector length. Default 2048.

    Returns:
        RDKit ExplicitBitVect if successful, None if mol is None or
        RDKit is unavailable.

    Notes:
        The caller is responsible for ensuring mol is valid (non-None,
        sanitized). This function does not re-sanitize.
    """
    if not _RDKIT_AVAILABLE:
        logger.error("morgan_fingerprint: RDKit not available")
        return None

    if mol is None:
        return None

    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=n_bits,
            useChirality=True,    # Include stereochemistry information
        )
        return fp
    except Exception as exc:
        logger.warning("morgan_fingerprint failed: %s", exc)
        return None


def morgan_fingerprint_from_smiles(
    smiles: str,
    radius: int = _DEFAULT_RADIUS,
    n_bits: int = _DEFAULT_N_BITS,
) -> Optional[object]:
    """
    Compute a Morgan fingerprint directly from a SMILES string.

    Convenience wrapper that handles SMILES parsing internally. Callers
    that already have a Mol object should prefer morgan_fingerprint().

    Args:
        smiles: SMILES string (canonical or non-canonical; will be parsed)
        radius: Morgan radius. Default 2 → ECFP4.
        n_bits: Fingerprint bit vector length. Default 2048.

    Returns:
        RDKit ExplicitBitVect if SMILES is valid, None on any failure.

    Examples:
        >>> fp = morgan_fingerprint_from_smiles("CCO")   # ethanol
        >>> fp is not None
        True
        >>> morgan_fingerprint_from_smiles("INVALID_XYZ")
        None
        >>> morgan_fingerprint_from_smiles("")
        None
    """
    if not _RDKIT_AVAILABLE:
        return None

    if not smiles or not smiles.strip():
        return None

    try:
        mol = Chem.MolFromSmiles(smiles.strip(), sanitize=True)
        if mol is None:
            return None
        return morgan_fingerprint(mol, radius=radius, n_bits=n_bits)
    except Exception as exc:
        logger.warning("morgan_fingerprint_from_smiles failed for %r: %s", smiles, exc)
        return None


# ---------------------------------------------------------------------------
# Similarity metric
# ---------------------------------------------------------------------------

def tanimoto_similarity(fp1, fp2) -> float:
    """
    Compute the Tanimoto (Jaccard) similarity between two fingerprints.

    Tanimoto similarity is the standard metric for comparing Morgan
    fingerprints in medicinal chemistry. It equals |A∩B| / |A∪B| for
    binary fingerprints.

    Args:
        fp1: RDKit ExplicitBitVect (from morgan_fingerprint or similar)
        fp2: RDKit ExplicitBitVect

    Returns:
        Float in [0.0, 1.0]:
            - 1.0 → identical fingerprints (same molecular scaffold)
            - 0.0 → no bits in common (structurally dissimilar)
        Returns 0.0 if either argument is None or RDKit is unavailable.

    Interpretation (ECFP4, per Maggiora et al. 2014):
        < 0.40  — Structurally dissimilar
        0.40-0.70 — Some similarity, likely different scaffold
        0.70-0.85 — Structural overlap, may share substitution patterns
        ≥ 0.85  — High similarity, probable same scaffold (IP review)
        ≥ 0.95  — Near-identical (enantiomer / prodrug / salt)
    """
    if not _RDKIT_AVAILABLE or fp1 is None or fp2 is None:
        return 0.0

    try:
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception as exc:
        logger.warning("tanimoto_similarity failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Serialization utilities
# ---------------------------------------------------------------------------

def fingerprint_to_hex(fp) -> Optional[str]:
    """
    Convert an RDKit ExplicitBitVect to a hex string for storage in AgentState.

    Enables fingerprint provenance storage without pickling RDKit objects.
    The hex string can be round-tripped back via hex_to_fingerprint().

    Args:
        fp: RDKit ExplicitBitVect

    Returns:
        Hex string representation, or None if fp is None.

    Example:
        >>> fp = morgan_fingerprint_from_smiles("CCO")
        >>> hex_str = fingerprint_to_hex(fp)
        >>> isinstance(hex_str, str) and len(hex_str) > 0
        True
    """
    if fp is None:
        return None

    try:
        # fp.ToBitString() → "00101...1" binary string
        # Convert to integer then hex for compact storage
        bit_str = fp.ToBitString()
        int_val = int(bit_str, 2)
        # Use zero-padding to preserve leading-zero bits
        n_bits = fp.GetNumBits()
        # Each 4 bits = 1 hex digit
        hex_digits = (n_bits + 3) // 4
        return format(int_val, f'0{hex_digits}x')
    except Exception as exc:
        logger.warning("fingerprint_to_hex failed: %s", exc)
        return None


def hex_to_fingerprint(hex_str: str, n_bits: int = _DEFAULT_N_BITS):
    """
    Reconstruct an RDKit ExplicitBitVect from a hex string.

    Inverse of fingerprint_to_hex(). Enables loading stored fingerprints
    from AgentState or databases without re-computing them.

    Args:
        hex_str: Hex string from fingerprint_to_hex()
        n_bits: Number of bits in the fingerprint (must match original)

    Returns:
        RDKit ExplicitBitVect, or None on failure.
    """
    if not _RDKIT_AVAILABLE or not hex_str:
        return None

    try:
        int_val = int(hex_str, 16)
        bit_str = format(int_val, f'0{n_bits}b')
        # Truncate or pad to exactly n_bits (handles rounding in hex_digits)
        bit_str = bit_str[-n_bits:].zfill(n_bits)

        fp = DataStructs.ExplicitBitVect(n_bits)
        for i, bit in enumerate(bit_str):
            if bit == '1':
                fp.SetBit(i)
        return fp
    except Exception as exc:
        logger.warning("hex_to_fingerprint failed: %s", exc)
        return None