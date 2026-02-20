"""
chemistry/sa_score.py

Core Synthetic Accessibility (SA) score calculation module for TRI_FLAG.

This module owns the algorithm layer — pure chemistry functions with no
dependency on agent infrastructure. Tools and analysis scripts import from here.

Algorithm: Ertl-Schuffenhauer (2009)
    Score = base_complexity
          + ring_complexity_penalty
          + stereochemistry_penalty
          + size_penalty
          - common_fragment_bonus

Scale: 1.0 (trivially easy) → 10.0 (practically impossible)

References:
    [1] Ertl, P. & Schuffenhauer, A. (2009). Estimation of synthetic
        accessibility score of drug-like molecules based on molecular
        complexity and fragment contributions. J. Cheminform., 1, 8.
        DOI: 10.1186/1758-2946-1-8

    [2] Bickerton, G.R. et al. (2012). Quantifying the chemical beauty
        of drugs. Nature Chemistry, 4, 90–98.
        DOI: 10.1038/nchem.1243

    [3] Gao, W. & Coley, C.W. (2020). The Synthesizability of Molecules
        Proposed by Generative Models. J. Chem. Inf. Model., 60, 5714–5723.
        DOI: 10.1021/acs.jcim.0c00174

    [4] Lovering, F. et al. (2009). Escape from flatland: increasing
        saturation as an approach to improving clinical success.
        J. Med. Chem., 52, 6752–6756.
        DOI: 10.1021/jm901241e
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import importlib
import importlib.util
import os
import sys as _sys

_sascorer = None
_SASCORER_AVAILABLE = False

# Method 1: Standard package import (works on some RDKit builds)
try:
    from rdkit.Contrib.SA_Score import sascorer as _sascorer
    _SASCORER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pass

# Method 2: Locate sascorer.py in RDKit's Contrib directory on disk
if not _SASCORER_AVAILABLE:
    try:
        import rdkit as _rdkit_pkg
        _rdkit_dir = os.path.dirname(_rdkit_pkg.__file__)
        _sascorer_candidates = [
            os.path.join(_rdkit_dir, "Contrib", "SA_Score", "sascorer.py"),
            os.path.join(_rdkit_dir, "Chem", "SA_Score", "sascorer.py"),
        ]
        for _path in _sascorer_candidates:
            if os.path.isfile(_path):
                _spec = importlib.util.spec_from_file_location("sascorer", _path)
                _sascorer = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_sascorer)
                _SASCORER_AVAILABLE = True
                break
    except Exception:
        pass

# Method 3: Search conda env site-packages for sascorer.py
if not _SASCORER_AVAILABLE:
    try:
        import glob
        import site
        _search_roots = []
        try:
            _search_roots += site.getsitepackages()
        except AttributeError:
            pass
        _search_roots += [_sys.prefix]
        for _root in _search_roots:
            _hits = glob.glob(
                os.path.join(_root, "**", "sascorer.py"),
                recursive=True
            )
            if _hits:
                _spec = importlib.util.spec_from_file_location("sascorer", _hits[0])
                _sascorer = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_sascorer)
                _SASCORER_AVAILABLE = True
                break
    except Exception:
        pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_sa_score(mol: Chem.Mol) -> float:
    """
    Calculate the Ertl-Schuffenhauer Synthetic Accessibility score.

    Args:
        mol: A valid, sanitized RDKit Mol object.

    Returns:
        SA score as a float in [1.0, 10.0].
        Lower scores indicate easier synthesis.

    Raises:
        RuntimeError: If sascorer is not available (see install note below).
        ValueError:   If mol is None.

    Install note:
        The sascorer is bundled with RDKit's Contrib module.
        Ensure you have: conda install -c conda-forge rdkit
    """
    if mol is None:
        raise ValueError("mol must not be None")
    if not _SASCORER_AVAILABLE:
        raise RuntimeError(
            "rdkit.Contrib.SA_Score not available. "
            "Install via: conda install -c conda-forge rdkit"
        )
    return float(_sascorer.calculateScore(mol))


def get_complexity_breakdown(mol: Chem.Mol) -> Dict[str, Any]:
    """
    Return a detailed breakdown of complexity contributors to the SA score.

    This provides explainability — useful for flagging *why* a molecule
    scores poorly, not just that it does.

    Args:
        mol: A valid, sanitized RDKit Mol object.

    Returns:
        Dictionary with the following keys:

        num_heavy_atoms (int):
            Total heavy atom count. Larger molecules are harder to make.
            Literature: Lipinski Ro5 ≤ 500 Da implies roughly ≤ 50 heavy atoms.

        num_rings (int):
            Total ring count. More rings → more complex synthesis.

        num_aromatic_rings (int):
            Aromatic ring count. Aromatic rings are easier than saturated ones
            (many commercial reagents), but fused systems raise complexity.

        num_saturated_rings (int):
            Non-aromatic (saturated/partially saturated) ring count.
            These are significantly harder to construct than aromatic rings [4].

        num_stereocenters (int):
            Stereocenters (R/S). Each doubles the number of possible isomers
            and typically requires chiral synthesis or resolution.

        num_spiro_atoms (int):
            Spiro-fused atoms. Highly unusual topology; very few commercial
            reagents contain spiro centers.

        num_bridgehead_atoms (int):
            Bridgehead atoms (bicyclic/polycyclic). Synthesis requires
            specialized methods (e.g., Diels-Alder, radical cyclization).

        max_ring_size (int):
            Largest ring size. Macrocycles (≥12 atoms) are notoriously
            difficult due to entropy and conformational flexibility.

        fraction_csp3 (float):
            Fraction of sp3 carbons (Fsp3). Higher Fsp3 correlates with
            better clinical success [4] but also with harder synthesis.
            Range [0, 1].

        has_macrocycle (bool):
            True if any ring has ≥12 atoms. Macrocycle synthesis is a
            specialized discipline with limited general methods.

        uncommon_atom_count (int):
            Count of heavy atoms that are not C, N, O, S, or halogens.
            Unusual atoms (Se, Te, B, Si, P in unusual valences, metals)
            require specialized reagents.

        warning_flags (List[str]):
            Human-readable list of synthesis challenges identified.
    """
    if mol is None:
        return _empty_breakdown()

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    num_heavy_atoms = mol.GetNumHeavyAtoms()
    num_rings = ring_info.NumRings()

    # Ring type counts
    num_aromatic_rings = sum(
        1 for ring in atom_rings if _is_aromatic_ring(mol, ring)
    )
    num_saturated_rings = num_rings - num_aromatic_rings

    # Stereocenters
    stereo_info = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    num_stereocenters = len(stereo_info)

    # Spiro and bridgehead
    num_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    num_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

    # Ring sizes
    ring_sizes = [len(ring) for ring in atom_rings] if atom_rings else [0]
    max_ring_size = max(ring_sizes) if ring_sizes else 0
    has_macrocycle = max_ring_size >= 12

    # Fsp3
    fraction_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)

    # Uncommon atoms
    common_atomic_nums = {6, 7, 8, 9, 15, 16, 17, 35, 53}  # C,N,O,F,P,S,Cl,Br,I
    uncommon_atom_count = sum(
        1 for atom in mol.GetAtoms()
        if atom.GetAtomicNum() not in common_atomic_nums
        and atom.GetAtomicNum() != 1  # exclude H
    )

    # Build warning flags
    warnings = _build_warning_flags(
        num_heavy_atoms=num_heavy_atoms,
        num_stereocenters=num_stereocenters,
        num_spiro=num_spiro,
        num_bridgehead=num_bridgehead,
        has_macrocycle=has_macrocycle,
        num_saturated_rings=num_saturated_rings,
        uncommon_atom_count=uncommon_atom_count,
        fraction_csp3=fraction_csp3,
    )

    return {
        "num_heavy_atoms": num_heavy_atoms,
        "num_rings": num_rings,
        "num_aromatic_rings": num_aromatic_rings,
        "num_saturated_rings": num_saturated_rings,
        "num_stereocenters": num_stereocenters,
        "num_spiro_atoms": num_spiro,
        "num_bridgehead_atoms": num_bridgehead,
        "max_ring_size": max_ring_size,
        "fraction_csp3": round(fraction_csp3, 3),
        "has_macrocycle": has_macrocycle,
        "uncommon_atom_count": uncommon_atom_count,
        "warning_flags": warnings,
    }


def sa_score_from_smiles(smiles: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Convenience wrapper: compute SA score directly from a SMILES string.

    Args:
        smiles: SMILES string (should be pre-validated).

    Returns:
        (sa_score, error_message)
        On success: (float, None)
        On failure: (None, str)
    """
    if not smiles or not smiles.strip():
        return None, "Empty SMILES string."

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, f"RDKit could not parse SMILES: '{smiles}'"

    try:
        score = calculate_sa_score(mol)
        return score, None
    except RuntimeError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"SA score computation failed: {exc}"


def full_sa_analysis(smiles: str) -> Dict[str, Any]:
    """
    Full SA analysis: score + complexity breakdown from a SMILES string.

    Returns a dict containing:
        success (bool)
        sa_score (float | None)
        complexity_breakdown (dict | None)
        error_message (str | None)
    """
    if not smiles or not smiles.strip():
        return {"success": False, "sa_score": None,
                "complexity_breakdown": None, "error_message": "Empty SMILES."}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"success": False, "sa_score": None, "complexity_breakdown": None,
                "error_message": f"Could not parse SMILES: '{smiles}'"}

    try:
        score = calculate_sa_score(mol)
        breakdown = get_complexity_breakdown(mol)
        return {
            "success": True,
            "sa_score": score,
            "complexity_breakdown": breakdown,
            "error_message": None,
        }
    except Exception as exc:
        return {"success": False, "sa_score": None,
                "complexity_breakdown": None, "error_message": str(exc)}


# ---------------------------------------------------------------------------
# Validation benchmarks
# ---------------------------------------------------------------------------

# Known molecules with published/expected SA scores for validation.
# Scores are approximate; exact values depend on RDKit version.
SA_SCORE_BENCHMARKS = {
    # Easy / simple
    "ethanol":          ("CCO",                             (1.0, 2.0)),
    "aspirin":          ("CC(=O)Oc1ccccc1C(=O)O",          (1.5, 2.5)),
    "ibuprofen":        ("CC(C)Cc1ccc(C(C)C(=O)O)cc1",     (1.8, 3.0)),
    "benzene":          ("c1ccccc1",                        (1.0, 2.0)),
    "paracetamol":      ("CC(=O)Nc1ccc(O)cc1",             (1.0, 2.5)),
    # Moderate
    "caffeine":         ("Cn1c(=O)c2c(ncn2C)n(c1=O)C",    (2.5, 4.0)),
    "nicotine":         ("CN1CCC[C@H]1c1cccnc1",           (2.5, 4.5)),
    "testosterone":     ("CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", (4.0, 6.5)),
    # Difficult / complex
    "taxol_scaffold":   (
        "O=C(O[C@@H]1C[C@]2(O)C(=O)[C@H](OC(=O)c3ccccc3)[C@@H]2[C@@H]1OC(C)=O)c1ccccc1",
        (6.5, 9.0)
    ),
    "cholesterol":      (
        "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
        (5.0, 8.0)
    ),
}


def validate_benchmarks() -> Dict[str, Any]:
    """
    Run SA score calculation on known benchmarks and report pass/fail.

    Returns a summary dict:
        passed (int), failed (int), results (List[dict])
    """
    results = []
    passed = failed = 0

    for name, (smiles, (lo, hi)) in SA_SCORE_BENCHMARKS.items():
        score, error = sa_score_from_smiles(smiles)
        in_range = (score is not None) and (lo <= score <= hi)
        status = "PASS" if in_range else "FAIL"
        if in_range:
            passed += 1
        else:
            failed += 1
        results.append({
            "name": name,
            "smiles": smiles,
            "sa_score": round(score, 2) if score else None,
            "expected_range": (lo, hi),
            "status": status,
            "error": error,
        })
        logger.debug("[benchmark] %s: score=%.2f range=[%.1f,%.1f] → %s",
                     name, score or -1, lo, hi, status)

    return {"passed": passed, "failed": failed, "results": results}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_aromatic_ring(mol: Chem.Mol, ring: Tuple[int, ...]) -> bool:
    """Return True if all atoms in the ring are aromatic."""
    return all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)


def _build_warning_flags(
    num_heavy_atoms: int,
    num_stereocenters: int,
    num_spiro: int,
    num_bridgehead: int,
    has_macrocycle: bool,
    num_saturated_rings: int,
    uncommon_atom_count: int,
    fraction_csp3: float,
) -> List[str]:
    """
    Generate human-readable synthesis challenge warnings.

    Thresholds are conservative — meant to flag for awareness, not discard.
    """
    flags = []

    if num_heavy_atoms > 50:
        flags.append(
            f"Large molecule ({num_heavy_atoms} heavy atoms): "
            "longer synthesis routes expected."
        )
    if num_stereocenters >= 3:
        flags.append(
            f"High stereocenter count ({num_stereocenters}): "
            "stereoselective synthesis required."
        )
    elif num_stereocenters >= 1:
        flags.append(
            f"{num_stereocenters} stereocenter(s): "
            "chiral synthesis or resolution needed."
        )
    if num_spiro > 0:
        flags.append(
            f"{num_spiro} spiro atom(s): "
            "spiro center construction requires specialized methods."
        )
    if num_bridgehead > 0:
        flags.append(
            f"{num_bridgehead} bridgehead atom(s): "
            "bicyclic/polycyclic framework — challenging ring construction."
        )
    if has_macrocycle:
        flags.append(
            "Macrocycle detected (ring ≥12 atoms): "
            "macrocycle synthesis is a specialized discipline."
        )
    if num_saturated_rings >= 3:
        flags.append(
            f"{num_saturated_rings} saturated rings: "
            "fewer commercial building blocks available vs aromatic systems."
        )
    if uncommon_atom_count > 0:
        flags.append(
            f"{uncommon_atom_count} uncommon heteroatom(s): "
            "specialized reagents required."
        )

    return flags


def _empty_breakdown() -> Dict[str, Any]:
    return {
        "num_heavy_atoms": 0,
        "num_rings": 0,
        "num_aromatic_rings": 0,
        "num_saturated_rings": 0,
        "num_stereocenters": 0,
        "num_spiro_atoms": 0,
        "num_bridgehead_atoms": 0,
        "max_ring_size": 0,
        "fraction_csp3": 0.0,
        "has_macrocycle": False,
        "uncommon_atom_count": 0,
        "warning_flags": [],
    }