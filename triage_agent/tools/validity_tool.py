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
from tools.base_tool import Tool
from agent.agent_state import AgentState
from molecule import Molecule


@dataclass
class ValidationResult:
    """
    Structured output from chemical validity checking.
    
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
            # Note: mol_object is NOT serialized
        }


class ValidityTool(Tool):
    """
    Chemical structure validation tool using RDKit.
    
    This tool performs comprehensive chemistry checks:
    1. SMILES string is non-empty and parseable
    2. Molecule can be constructed by RDKit
    3. Molecule passes RDKit's sanitization (aromaticity, valence, etc.)
    4. Molecule has at least one atom (non-empty structure)
    
    Usage:
        >>> tool = ValidityTool()
        >>> # Tool is called by agent with: tool.run(state)
    """
    
    def __init__(self):
        """Initialize the validity checking tool."""
        # Set the name attribute required by base class
        self.name = "ValidityTool"
        self.description = "Validates chemical structure using RDKit sanitization"
        self.logger = logging.getLogger(__name__)
        
        if not RDKIT_AVAILABLE:
            self.logger.error(
                "RDKit is not installed. ValidityTool will mark all molecules as invalid. "
                "Install with: conda install -c conda-forge rdkit"
            )
    
    def run(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute validity checking on the molecule in the current state.
        
        This is the main entry point called by TriageAgent during tool execution.
        
        Args:
            state: AgentState containing the molecule to validate
        
        Returns:
            Dictionary with validation results:
            {
                'tool_name': 'ValidityTool',
                'is_valid': bool,
                'error_message': str or None,
                'smiles_canonical': str or None,
                'num_atoms': int,
                'num_bonds': int,
                'molecule_id': str
            }
        """
        self.logger.info(f"{self.name}: Validating molecule {state.molecule_id}")
        
        # Extract molecule information from state
        raw_input = state.raw_input
        
        # Construct Molecule object from raw_input
        try:
            if isinstance(raw_input, Molecule):
                molecule = raw_input
            elif isinstance(raw_input, dict):
                molecule = Molecule(
                    molecule_id=state.molecule_id,
                    smiles=raw_input.get('smiles', ''),
                    name=raw_input.get('name'),
                    metadata=raw_input.get('metadata', {})
                )
            elif isinstance(raw_input, str):
                # Assume it's a SMILES string
                molecule = Molecule(
                    molecule_id=state.molecule_id,
                    smiles=raw_input
                )
            else:
                # Unknown input type
                return {
                    'tool_name': self.name,
                    'molecule_id': state.molecule_id,
                    'is_valid': False,
                    'error_message': f'Unknown input type: {type(raw_input)}',
                    'smiles_canonical': None,
                    'num_atoms': 0,
                    'num_bonds': 0
                }
        except Exception as e:
            # Failed to construct Molecule object
            return {
                'tool_name': self.name,
                'molecule_id': state.molecule_id,
                'is_valid': False,
                'error_message': f'Failed to construct molecule: {str(e)}',
                'smiles_canonical': None,
                'num_atoms': 0,
                'num_bonds': 0
            }
        
        # Validate the molecule
        return self._validate_molecule(molecule)
    
    def _validate_molecule(self, molecule: Molecule) -> Dict[str, Any]:
        """
        Validate a molecule's chemical structure using RDKit.
        
        Args:
            molecule: Molecule object to validate
        
        Returns:
            Dictionary with validation results
        """
        # Step 1: Check if RDKit is available
        if not RDKIT_AVAILABLE:
            error_msg = "RDKit not available - cannot validate molecule"
            self.logger.error(f"{self.name}: {error_msg}")
            validation_result = ValidationResult(
                is_valid=False,
                error_message=error_msg
            )
            return self._format_result(molecule, validation_result)
        
        # Step 2: Check if SMILES string exists and is non-empty
        if not molecule.smiles or not molecule.smiles.strip():
            error_msg = "SMILES string is empty or whitespace-only"
            self.logger.warning(f"{self.name}: {error_msg} for {molecule.molecule_id}")
            validation_result = ValidationResult(
                is_valid=False,
                error_message=error_msg
            )
            return self._format_result(molecule, validation_result)
        
        # Step 3: Attempt to parse SMILES with RDKit
        try:
            mol = Chem.MolFromSmiles(molecule.smiles, sanitize=True)
            
            if mol is None:
                error_msg = f"RDKit failed to parse SMILES: '{molecule.smiles}'"
                self.logger.warning(f"{self.name}: {error_msg} for {molecule.molecule_id}")
                validation_result = ValidationResult(
                    is_valid=False,
                    error_message=error_msg
                )
                return self._format_result(molecule, validation_result)
            
        except Exception as e:
            error_msg = f"RDKit exception during parsing: {str(e)}"
            self.logger.error(
                f"{self.name}: {error_msg} for {molecule.molecule_id}",
                exc_info=True
            )
            validation_result = ValidationResult(
                is_valid=False,
                error_message=error_msg
            )
            return self._format_result(molecule, validation_result)
        
        # Step 4: Validate molecule properties
        try:
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            
            if num_atoms == 0:
                error_msg = "Molecule has zero atoms (empty structure)"
                self.logger.warning(f"{self.name}: {error_msg} for {molecule.molecule_id}")
                validation_result = ValidationResult(
                    is_valid=False,
                    error_message=error_msg
                )
                return self._format_result(molecule, validation_result)
            
            # Generate canonical SMILES
            smiles_canonical = Chem.MolToSmiles(mol, canonical=True)
            
        except Exception as e:
            error_msg = f"Failed to extract molecular properties: {str(e)}"
            self.logger.error(
                f"{self.name}: {error_msg} for {molecule.molecule_id}",
                exc_info=True
            )
            validation_result = ValidationResult(
                is_valid=False,
                error_message=error_msg
            )
            return self._format_result(molecule, validation_result)
        
        # Step 5: Molecule is VALID
        self.logger.info(
            f"{self.name}: Molecule {molecule.molecule_id} is VALID "
            f"({num_atoms} atoms, {num_bonds} bonds, canonical: {smiles_canonical})"
        )
        
        validation_result = ValidationResult(
            is_valid=True,
            mol_object=mol,
            error_message=None,
            smiles_canonical=smiles_canonical,
            num_atoms=num_atoms,
            num_bonds=num_bonds
        )
        
        return self._format_result(molecule, validation_result)
    
    def _format_result(
        self,
        molecule: Molecule,
        validation_result: ValidationResult
    ) -> Dict[str, Any]:
        """
        Format ValidationResult into the standard tool output format.
        
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
# Helper Functions
# =============================================================================

def validate_smiles(smiles: str) -> tuple:
    """
    Standalone function to quickly validate a SMILES string.
    
    Args:
        smiles: SMILES string to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    
    Example:
        >>> is_valid, error = validate_smiles("CCO")
        >>> is_valid
        True
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