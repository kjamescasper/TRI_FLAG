"""
molecule.py

Molecule data structure for representing chemical compounds in TRI_FLAG.

This module provides an immutable data container for molecules being evaluated
by the triage system. It ensures consistent data representation across all tools
and components.

Week: 2-3 (Foundation)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class Molecule:
    """
    Immutable representation of a chemical compound.
    
    This class serves as the primary data container for molecules being evaluated
    by the TRI_FLAG system. It stores the minimal required information to uniquely
    identify and represent a chemical structure.
    
    The Molecule is designed to be:
    - Immutable (frozen=True): Once created, cannot be modified
    - Hashable: Can be used as dictionary keys or in sets
    - Validated: Construction enforces non-empty required fields
    
    Attributes:
        molecule_id: Unique identifier for this molecule
                     Examples: "MOL_001", "CHEMBL123", "TEST_COMPOUND_042"
        
        smiles: SMILES (Simplified Molecular Input Line Entry System) string
                Canonical structure representation of the molecule
                Examples: "CCO" (ethanol), "c1ccccc1" (benzene)
        
        name: Optional human-readable name for the molecule
              Examples: "Aspirin", "Caffeine", "Compound_A"
        
        metadata: Optional dictionary for additional properties
                  Examples: {"source": "ChEMBL", "batch": "2024-02"}
    
    Examples:
        >>> # Basic molecule
        >>> mol = Molecule(molecule_id="MOL_001", smiles="CCO")
        >>> mol.molecule_id
        'MOL_001'
        >>> mol.smiles
        'CCO'
        
        >>> # Molecule with name and metadata
        >>> aspirin = Molecule(
        ...     molecule_id="CHEMBL25",
        ...     smiles="CC(=O)Oc1ccccc1C(=O)O",
        ...     name="Aspirin",
        ...     metadata={"source": "ChEMBL", "mw": 180.16}
        ... )
    """
    
    # Required fields
    molecule_id: str
    smiles: str
    
    # Optional fields
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """
        Validation executed after initialization.
        
        Enforces basic data integrity constraints. Since this is a frozen
        dataclass, validation must happen in __post_init__ before the
        object becomes immutable.
        
        Raises:
            ValueError: If molecule_id is empty or whitespace-only
            ValueError: If smiles is empty or whitespace-only
        """
        # Validate molecule_id
        if not self.molecule_id or not self.molecule_id.strip():
            raise ValueError("molecule_id cannot be empty or whitespace")
        
        # Validate SMILES
        if not self.smiles or not self.smiles.strip():
            raise ValueError("smiles cannot be empty or whitespace")
        
        # Note: We do NOT validate chemical correctness here
        # That's the job of ValidityTool (which uses RDKit)
        # This validation only ensures non-empty strings
        
        # Ensure metadata is a dictionary
        if not isinstance(self.metadata, dict):
            object.__setattr__(self, 'metadata', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Molecule to dictionary for serialization.
        
        Useful for:
        - JSON serialization for API responses
        - Database storage
        - Logging and debugging
        
        Returns:
            Dictionary representation with all fields
        
        Example:
            >>> mol = Molecule(molecule_id="MOL_001", smiles="CCO", name="Ethanol")
            >>> mol.to_dict()
            {
                'molecule_id': 'MOL_001',
                'smiles': 'CCO',
                'name': 'Ethanol',
                'metadata': {}
            }
        """
        return {
            'molecule_id': self.molecule_id,
            'smiles': self.smiles,
            'name': self.name,
            'metadata': self.metadata.copy()  # Return copy to prevent mutation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Molecule':
        """
        Reconstruct Molecule from dictionary.
        
        Inverse of to_dict(). Enables deserialization from storage/API.
        
        Args:
            data: Dictionary with keys: molecule_id, smiles, 
                  and optionally name, metadata
        
        Returns:
            Molecule object
        
        Raises:
            KeyError: If required keys (molecule_id, smiles) are missing
            ValueError: If data fails validation in __post_init__
        
        Example:
            >>> data = {
            ...     'molecule_id': 'MOL_001',
            ...     'smiles': 'CCO',
            ...     'name': 'Ethanol'
            ... }
            >>> mol = Molecule.from_dict(data)
            >>> mol.molecule_id
            'MOL_001'
        """
        return cls(
            molecule_id=data['molecule_id'],
            smiles=data['smiles'],
            name=data.get('name'),
            metadata=data.get('metadata', {})
        )
    
    def __str__(self) -> str:
        """
        Human-readable string representation.
        
        Example:
            >>> mol = Molecule(molecule_id="MOL_001", smiles="CCO", name="Ethanol")
            >>> str(mol)
            'Molecule(MOL_001: Ethanol)'
            
            >>> mol_no_name = Molecule(molecule_id="MOL_002", smiles="CC")
            >>> str(mol_no_name)
            'Molecule(MOL_002)'
        """
        if self.name:
            return f"Molecule({self.molecule_id}: {self.name})"
        return f"Molecule({self.molecule_id})"
    
    def __repr__(self) -> str:
        """
        Developer-friendly representation showing all fields.
        
        Example:
            >>> mol = Molecule(molecule_id="MOL_001", smiles="CCO")
            >>> repr(mol)
            "Molecule(molecule_id='MOL_001', smiles='CCO', name=None, metadata={})"
        """
        return (
            f"Molecule(molecule_id={self.molecule_id!r}, "
            f"smiles={self.smiles!r}, "
            f"name={self.name!r}, "
            f"metadata={self.metadata!r})"
        )


# =============================================================================
# Convenience Factory Function
# =============================================================================

def create_molecule(
    molecule_id: str,
    smiles: str,
    name: Optional[str] = None,
    **metadata_kwargs
) -> Molecule:
    """
    Convenience factory for creating molecules with keyword metadata.
    
    This allows metadata to be passed as keyword arguments rather than
    requiring explicit dictionary construction.
    
    Args:
        molecule_id: Unique identifier
        smiles: SMILES structure representation
        name: Optional human-readable name
        **metadata_kwargs: Additional metadata as keyword arguments
    
    Returns:
        Molecule object
    
    Example:
        >>> # Instead of:
        >>> mol = Molecule(
        ...     molecule_id="MOL_001",
        ...     smiles="CCO",
        ...     metadata={"mw": 46.07, "source": "ChEMBL"}
        ... )
        
        >>> # You can write:
        >>> mol = create_molecule(
        ...     molecule_id="MOL_001",
        ...     smiles="CCO",
        ...     mw=46.07,
        ...     source="ChEMBL"
        ... )
    """
    return Molecule(
        molecule_id=molecule_id,
        smiles=smiles,
        name=name,
        metadata=metadata_kwargs
    )


# =============================================================================
# Design Notes
# =============================================================================
#
# Why frozen dataclass?
# ---------------------
# - Immutability prevents accidental modification during processing
# - Hashable molecules can be used as dict keys or set members
# - Thread-safe by design (important for future parallel processing)
# - Clear semantics: if you need different properties, create new molecule
#
# Why minimal required fields?
# ----------------------------
# - molecule_id: Essential for tracking and logging
# - smiles: Essential for chemical structure representation
# - name: Optional (many workflows use IDs only)
# - metadata: Extensible without code changes
#
# Why NOT include computed properties?
# -------------------------------------
# - Computed properties (molecular weight, SA score, etc.) belong in tools
# - Tool results are stored in AgentState.tool_results, not in Molecule
# - Keeps Molecule as pure data container
# - Avoids coupling to specific chemistry libraries
#
# Integration with TRI_FLAG:
# --------------------------
# - TriageAgent receives/constructs Molecule objects from raw_input
# - ValidityTool and other tools receive Molecule as input
# - Tools store results in AgentState, not in Molecule
# - Molecule remains unchanged throughout the pipeline
#