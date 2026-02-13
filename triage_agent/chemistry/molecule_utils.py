"""
chemistry/molecule_utils.py

Utility functions for molecular operations and conversions.

This module provides chemistry-specific helper functions that are used
across multiple tools and components. It serves as a shared library
for common molecular operations.

Week: 3 (extended from Week 2 scaffolding)
"""

import logging
from typing import Optional, Tuple, List, Dict, Any

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from ..molecule import Molecule
from ..tools.validity_tool import validate_smiles


logger = logging.getLogger(__name__)


# =============================================================================
# Molecule Conversion Functions
# =============================================================================

def smiles_to_mol(smiles: str, sanitize: bool = True) -> Optional[Any]:
    """
    Convert SMILES string to RDKit Mol object.
    
    Args:
        smiles: SMILES representation
        sanitize: Whether to perform sanitization (default: True)
    
    Returns:
        RDKit Mol object if successful, None if parsing fails
    
    Example:
        >>> mol = smiles_to_mol("CCO")
        >>> mol is not None
        True
        >>> invalid = smiles_to_mol("C(C)(C)(C)(C)C")  # Invalid valence
        >>> invalid is None
        True
    """
    if not RDKIT_AVAILABLE:
        logger.error("RDKit not available")
        return None
    
    if not smiles or not smiles.strip():
        logger.warning("Empty SMILES string provided")
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        return mol
    except Exception as e:
        logger.warning(f"Failed to parse SMILES '{smiles}': {e}")
        return None


def mol_to_canonical_smiles(mol: Any) -> Optional[str]:
    """
    Convert RDKit Mol object to canonical SMILES.
    
    Canonical SMILES ensures that different representations of the
    same molecule (e.g., "CCO" vs "OCC") are standardized.
    
    Args:
        mol: RDKit Mol object
    
    Returns:
        Canonical SMILES string, or None if conversion fails
    
    Example:
        >>> mol = smiles_to_mol("OCC")  # Reverse atom order
        >>> canonical = mol_to_canonical_smiles(mol)
        >>> canonical
        "CCO"  # Standardized form
    """
    if not RDKIT_AVAILABLE:
        logger.error("RDKit not available")
        return None
    
    if mol is None:
        logger.warning("Cannot convert None to SMILES")
        return None
    
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        logger.error(f"Failed to convert mol to SMILES: {e}")
        return None


def canonicalize_smiles(smiles: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Canonicalize a SMILES string (validate + standardize).
    
    This is a convenience function that combines validation and canonicalization.
    
    Args:
        smiles: Input SMILES string
    
    Returns:
        Tuple of (success, canonical_smiles, error_message)
        - success: True if canonicalization succeeded
        - canonical_smiles: Standardized SMILES if successful, None otherwise
        - error_message: None if successful, error description otherwise
    
    Example:
        >>> success, canon, error = canonicalize_smiles("OCC")
        >>> success
        True
        >>> canon
        "CCO"
        >>> success, canon, error = canonicalize_smiles("invalid")
        >>> success
        False
    """
    mol = smiles_to_mol(smiles, sanitize=True)
    
    if mol is None:
        return False, None, f"Failed to parse SMILES: {smiles}"
    
    canonical = mol_to_canonical_smiles(mol)
    
    if canonical is None:
        return False, None, "Failed to generate canonical SMILES"
    
    return True, canonical, None


# =============================================================================
# Molecule Property Extraction
# =============================================================================

def get_basic_properties(mol: Any) -> Dict[str, Any]:
    """
    Extract basic molecular properties from RDKit Mol object.
    
    Args:
        mol: RDKit Mol object
    
    Returns:
        Dictionary of properties:
        {
            'num_atoms': int,
            'num_bonds': int,
            'num_heavy_atoms': int,
            'molecular_weight': float,
            'num_rotatable_bonds': int,
            'num_aromatic_rings': int
        }
    
    Example:
        >>> mol = smiles_to_mol("CCO")
        >>> props = get_basic_properties(mol)
        >>> props['num_atoms']
        9  # Includes hydrogens
        >>> props['num_heavy_atoms']
        3  # C, C, O only
    """
    if not RDKIT_AVAILABLE or mol is None:
        return {}
    
    try:
        props = {
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'num_heavy_atoms': mol.GetNumHeavyAtoms(),
            'molecular_weight': Descriptors.MolWt(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'num_aromatic_rings': Descriptors.NumAromaticRings(mol)
        }
        return props
    except Exception as e:
        logger.error(f"Failed to extract properties: {e}")
        return {}


# =============================================================================
# Batch Operations
# =============================================================================

def validate_molecule_batch(molecules: List[Molecule]) -> Dict[str, Any]:
    """
    Validate a batch of molecules and return summary statistics.
    
    Useful for:
    - Preprocessing large datasets
    - Quality control reports
    - Filtering invalid molecules from input files
    
    Args:
        molecules: List of Molecule objects to validate
    
    Returns:
        Dictionary with statistics:
        {
            'total': int,
            'valid': int,
            'invalid': int,
            'valid_ids': List[str],
            'invalid_ids': List[str],
            'errors': Dict[str, str]  # molecule_id -> error_message
        }
    
    Example:
        >>> mols = [
        ...     Molecule("MOL_001", "CCO"),
        ...     Molecule("MOL_002", "invalid"),
        ...     Molecule("MOL_003", "c1ccccc1")
        ... ]
        >>> stats = validate_molecule_batch(mols)
        >>> stats['valid']
        2
        >>> stats['invalid']
        1
    """
    results = {
        'total': len(molecules),
        'valid': 0,
        'invalid': 0,
        'valid_ids': [],
        'invalid_ids': [],
        'errors': {}
    }
    
    for mol in molecules:
        # Use the standalone validation function
        from tools.validity_tool import validate_smiles
        is_valid, error = validate_smiles(mol.smiles)
        
        if is_valid:
            results['valid'] += 1
            results['valid_ids'].append(mol.molecule_id)
        else:
            results['invalid'] += 1
            results['invalid_ids'].append(mol.molecule_id)
            results['errors'][mol.molecule_id] = error
    
    return results


# =============================================================================
# Design Notes
# =============================================================================
#
# Why separate molecule_utils from validity_tool?
# -----------------------------------------------
# - ValidityTool is agent-specific (implements Tool interface)
# - molecule_utils is general-purpose (usable anywhere)
# - Prevents circular dependencies
# - Enables standalone testing of conversion functions
#
# These utilities will be extended in future weeks for:
# - SA score calculation (Week 4)
# - Fingerprint generation for similarity (Week 5)
# - Molecular descriptor calculation (Week 6+)
#