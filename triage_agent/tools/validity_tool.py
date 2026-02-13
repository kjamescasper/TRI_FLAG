"""
tools/validity_tool.py

Chemical validity checking tool using RDKit sanitization.

This tool serves as the first line of defense in the triage pipeline,
rejecting chemically invalid or malformed structures before they reach
computationally expensive analysis tools.

Design Philosophy:
    - Fail fast: Invalid molecules stop the pipeline early
    - Explicit errors: Every failure has a clear, logged reason
    - Scientific rigor: Uses RDKit's sanitization as ground truth
    - No silent failures: All edge cases are handled and logged

Week: 3
Priority: CRITICAL (blocking for all downstream tools)
"""

import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass

# RDKit imports - core chemistry library
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available - ValidityTool will fail gracefully")

# Internal imports
from .base_tool import Tool
from ..molecule import Molecule


@dataclass
class ValidationResult:
    """
    Structured output from chemical validity checking.
    
    This encapsulates all information about whether a molecule is valid,
    why it might be invalid, and what the valid molecular object is.
    
    Attributes:
        is_valid: Boolean indicating if molecule passed validation
        mol_object: RDKit Mol object if valid, None if invalid
        error_message: Specific reason for failure if invalid
        smiles_canonical: Canonicalized SMILES if valid, None if invalid
        num_atoms: Atom count if valid, 0 if invalid
        num_bonds: Bond count if valid, 0 if invalid
    """
    is_valid: bool
    mol_object: Optional[Any] = None  # RDKit Mol object
    error_message: Optional[str] = None
    smiles_canonical: Optional[str] = None
    num_atoms: int = 0
    num_bonds: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage in AgentState."""
        return {
            'is_valid': self.is_valid,
            'error_message': self.error_message,
            'smiles_canonical': self.smiles_canonical,
            'num_atoms': self.num_atoms,
            'num_bonds': self.num_bonds
            # Note: mol_object is NOT serialized (RDKit object can't be JSON serialized)
        }


class ValidityTool(Tool):
    """
    Chemical structure validation tool using RDKit.
    
    This tool performs the following checks:
    1. SMILES string is non-empty and parseable
    2. Molecule can be constructed by RDKit
    3. Molecule passes RDKit's sanitization (aromaticity, valence, etc.)
    4. Molecule has at least one atom (non-empty structure)
    
    Validation Flow:
        Input SMILES → Parse to Mol → Sanitize → Extract properties → Result
                 ↓           ↓            ↓
              FAIL        FAIL         FAIL
                 ↓           ↓            ↓
             Log reason  Log reason   Log reason
    
    Scientific Background:
        RDKit's sanitization checks:
        - Valence correctness (e.g., carbon has max 4 bonds)
        - Aromaticity perception (benzene rings, etc.)
        - Radical/charge consistency
        - Stereochemistry validity
        
        A molecule that fails sanitization is chemically implausible
        and should not proceed to downstream analysis.
    
    Usage:
        >>> tool = ValidityTool()
        >>> molecule = Molecule(molecule_id="MOL_001", smiles="CCO")
        >>> result = tool.evaluate(molecule, context={})
        >>> result['is_valid']
        True
    """
    
    def __init__(self):
        """
        Initialize the validity checking tool.
        
        Raises:
            ImportError: If RDKit is not available (fails gracefully with logging)
        """
        super().__init__(
            name="ValidityTool",
            description="Validates chemical structure using RDKit sanitization"
        )
        self.logger = logging.getLogger(__name__)
        
        # Check if RDKit is available
        if not RDKIT_AVAILABLE:
            self.logger.error(
                "RDKit is not installed. ValidityTool will mark all molecules as invalid. "
                "Install with: pip install rdkit"
            )
    
    def evaluate(self, candidate: Molecule, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a molecule's chemical structure.
        
        This is the main entry point called by TriageAgent during tool execution.
        
        Args:
            candidate: Molecule object to validate
            context: Additional context (unused for validity checking, but required by Tool interface)
        
        Returns:
            Dictionary containing ValidationResult data plus metadata:
            {
                'tool_name': 'ValidityTool',
                'is_valid': bool,
                'error_message': str or None,
                'smiles_canonical': str or None,
                'num_atoms': int,
                'num_bonds': int,
                'molecule_id': str  # For traceability
            }
        
        Notes:
            - This method never raises exceptions
            - All failures are caught and returned as is_valid=False
            - Detailed error messages are logged and returned
        
        Scientific Rationale:
            We use RDKit's Chem.MolFromSmiles with sanitize=True because:
            1. It's the gold standard in cheminformatics
            2. Sanitization catches >95% of invalid structures
            3. It's computationally cheap (< 1ms per molecule)
            4. Failures provide actionable error messages
        """
        self.logger.info(f"{self.name}: Validating molecule {candidate.molecule_id}")
        
        # Step 1: Pre-validation checks (before RDKit)
        # -----------------------------------------------
        # Check if RDKit is available
        if not RDKIT_AVAILABLE:
            error_msg = "RDKit not available - cannot validate molecule"
            self.logger.error(f"{self.name}: {error_msg}")
            validation_result = ValidationResult(
                is_valid=False,
                error_message=error_msg
            )
            return self._format_result(candidate, validation_result)
        
        # Check if SMILES string exists and is non-empty
        if not candidate.smiles or not candidate.smiles.strip():
            error_msg = "SMILES string is empty or whitespace-only"
            self.logger.warning(f"{self.name}: {error_msg} for {candidate.molecule_id}")
            validation_result = ValidationResult(
                is_valid=False,
                error_message=error_msg
            )
            return self._format_result(candidate, validation_result)
        
        # Step 2: Attempt to parse SMILES with RDKit
        # -------------------------------------------
        try:
            # MolFromSmiles with sanitize=True performs:
            # 1. SMILES parsing
            # 2. Aromaticity perception
            # 3. Valence checking
            # 4. Kekulization
            # 5. Radical/charge validation
            mol = Chem.MolFromSmiles(candidate.smiles, sanitize=True)
            
            # RDKit returns None for invalid molecules (not an exception)
            if mol is None:
                error_msg = f"RDKit failed to parse SMILES: '{candidate.smiles}'"
                self.logger.warning(
                    f"{self.name}: {error_msg} for {candidate.molecule_id}"
                )
                validation_result = ValidationResult(
                    is_valid=False,
                    error_message=error_msg
                )
                return self._format_result(candidate, validation_result)
            
        except Exception as e:
            # Catch any unexpected exceptions from RDKit
            # (rare, but possible with malformed input)
            error_msg = f"RDKit exception during parsing: {str(e)}"
            self.logger.error(
                f"{self.name}: {error_msg} for {candidate.molecule_id}",
                exc_info=True  # Log full stack trace
            )
            validation_result = ValidationResult(
                is_valid=False,
                error_message=error_msg
            )
            return self._format_result(candidate, validation_result)
        
        # Step 3: Validate molecule properties
        # -------------------------------------
        try:
            # Extract basic molecular properties
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            
            # Check for empty molecule (edge case: mol exists but has 0 atoms)
            if num_atoms == 0:
                error_msg = "Molecule has zero atoms (empty structure)"
                self.logger.warning(
                    f"{self.name}: {error_msg} for {candidate.molecule_id}"
                )
                validation_result = ValidationResult(
                    is_valid=False,
                    error_message=error_msg
                )
                return self._format_result(candidate, validation_result)
            
            # Generate canonical SMILES for standardization
            # This ensures "CCO" and "OCC" are treated as the same molecule
            smiles_canonical = Chem.MolToSmiles(mol, canonical=True)
            
        except Exception as e:
            # Property extraction failed (very rare)
            error_msg = f"Failed to extract molecular properties: {str(e)}"
            self.logger.error(
                f"{self.name}: {error_msg} for {candidate.molecule_id}",
                exc_info=True
            )
            validation_result = ValidationResult(
                is_valid=False,
                error_message=error_msg
            )
            return self._format_result(candidate, validation_result)
        
        # Step 4: Molecule is VALID
        # -------------------------
        self.logger.info(
            f"{self.name}: Molecule {candidate.molecule_id} is VALID "
            f"({num_atoms} atoms, {num_bonds} bonds, canonical: {smiles_canonical})"
        )
        
        validation_result = ValidationResult(
            is_valid=True,
            mol_object=mol,  # Store for potential downstream use
            error_message=None,
            smiles_canonical=smiles_canonical,
            num_atoms=num_atoms,
            num_bonds=num_bonds
        )
        
        return self._format_result(candidate, validation_result)
    
    def _format_result(
        self,
        molecule: Molecule,
        validation_result: ValidationResult
    ) -> Dict[str, Any]:
        """
        Format ValidationResult into the standard tool output format.
        
        This ensures consistency with the Tool interface and makes results
        easy to store in AgentState.tool_results.
        
        Args:
            molecule: The molecule that was validated
            validation_result: Validation outcome
        
        Returns:
            Dictionary with tool metadata and validation data
        """
        result = {
            'tool_name': self.name,
            'molecule_id': molecule.molecule_id,
            **validation_result.to_dict()
        }
        return result


# =============================================================================
# Helper Functions for External Use
# =============================================================================

def validate_smiles(smiles: str) -> tuple[bool, Optional[str]]:
    """
    Standalone function to quickly validate a SMILES string.
    
    Useful for:
    - Pre-validation before creating Molecule objects
    - Quick checks in data preprocessing pipelines
    - Testing/debugging
    
    Args:
        smiles: SMILES string to validate
    
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if SMILES is chemically valid
        - error_message: None if valid, error description if invalid
    
    Example:
        >>> is_valid, error = validate_smiles("CCO")
        >>> is_valid
        True
        >>> is_valid, error = validate_smiles("C(C)(C)(C)(C)C")  # Too many bonds on carbon
        >>> is_valid
        False
        >>> error
        "RDKit failed to parse SMILES: ..."
    """
    if not RDKIT_AVAILABLE:
        return False, "RDKit not available"
    
    if not smiles or not smiles.strip():
        return False, "SMILES string is empty"
    
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return False, f"RDKit failed to parse SMILES: '{smiles}'"
        
        if mol.GetNumAtoms() == 0:
            return False, "Molecule has zero atoms"
        
        return True, None
    
    except Exception as e:
        return False, f"RDKit exception: {str(e)}"


# =============================================================================
# Design Notes
# =============================================================================
#
# Why RDKit sanitization is the right choice:
# -------------------------------------------
# 1. Industry standard: Used by ChEMBL, PubChem, all major pharma companies
# 2. Comprehensive: Checks valence, aromaticity, stereochemistry, radicals
# 3. Fast: < 1ms per molecule, can validate millions per day
# 4. Well-tested: 15+ years of development, handles edge cases
# 5. Informative errors: Tells you WHY a molecule is invalid
#
# Alternative approaches considered and rejected:
# ----------------------------------------------
# - SMILES regex validation: Too simplistic, misses chemical invalidity
# - OpenBabel: Less comprehensive sanitization than RDKit
# - Custom valence checking: Reinventing the wheel, prone to bugs
#
# Error handling philosophy:
# -------------------------
# - NEVER raise exceptions from evaluate()
# - ALL failures return is_valid=False with explanation
# - This prevents invalid molecules from crashing the pipeline
# - Enables bulk processing of mixed valid/invalid datasets
#
# Integration with agent workflow:
# --------------------------------
# - ValidityTool should be registered FIRST in agent.tools list
# - Agent should check tool_results['ValidityTool']['is_valid']
# - If False, agent should terminate early (no need to run other tools)
# - PolicyEngine should automatically DISCARD invalid molecules
#