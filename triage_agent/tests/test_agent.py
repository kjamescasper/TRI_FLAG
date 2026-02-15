"""
tests/test_agent.py

Unit tests for Week 3: Chemical Validity Checking

Tests cover:
1. Valid SMILES handling
2. Invalid SMILES rejection
3. Edge cases (empty, None, malformed)
4. Integration with agent workflow
"""

import pytest
import logging
from molecule import Molecule
from tools.validity_tool import ValidityTool, validate_smiles
from agent.triage_agent import TriageAgent
from policies.policy_engine import PolicyEngine
from agent.agent_state import AgentState


# =============================================================================
# Unit Tests for ValidityTool
# =============================================================================

class TestValidityTool:
    """Test suite for ValidityTool functionality."""
    
    def setup_method(self):
        """Initialize tool before each test."""
        self.tool = ValidityTool()
    
    def test_valid_simple_molecule(self):
        """Test validation of a simple valid molecule (ethanol)."""
        mol = Molecule(molecule_id="TEST_001", smiles="CCO")
        result = self.tool._validate_molecule(mol)
        
        assert result['is_valid'] is True
        assert result['error_message'] is None
        assert result['smiles_canonical'] == "CCO"
        assert result['num_atoms'] > 0
        assert result['num_bonds'] > 0
    
    def test_valid_aromatic_molecule(self):
        """Test validation of an aromatic molecule (benzene)."""
        mol = Molecule(molecule_id="TEST_002", smiles="c1ccccc1")
        result = self.tool._validate_molecule(mol)
        
        assert result['is_valid'] is True
        assert result['smiles_canonical'] == "c1ccccc1"
        assert result['num_atoms'] == 6  # 6 carbons
    
    def test_invalid_valence(self):
        """Test rejection of molecule with invalid valence."""
        # Carbon with 5 bonds (chemically impossible)
        mol = Molecule(molecule_id="TEST_003", smiles="C(C)(C)(C)(C)C")
        result = self.tool._validate_molecule(mol)
        
        assert result['is_valid'] is False
        assert result['error_message'] is not None
        assert "RDKit failed to parse" in result['error_message']
    
    def test_empty_smiles(self):
        """Test rejection of empty SMILES string."""
        # Can't create Molecule with empty SMILES (validation rejects it)
        # So test via the standalone validate_smiles function
        is_valid, error = validate_smiles("")
        assert is_valid is False
        assert "empty" in error.lower()
    
    def test_whitespace_only_smiles(self):
        """Test rejection of whitespace-only SMILES."""
        # Can't create Molecule with whitespace SMILES (validation rejects it)
        # So test via the standalone validate_smiles function
        is_valid, error = validate_smiles("   ")
        assert is_valid is False
        assert "empty" in error.lower()
    
    def test_malformed_smiles(self):
        """Test rejection of syntactically invalid SMILES."""
        mol = Molecule(molecule_id="TEST_006", smiles="C(((")
        result = self.tool._validate_molecule(mol)
        
        assert result['is_valid'] is False
        assert result['error_message'] is not None
    
    def test_canonicalization(self):
        """Test that non-canonical SMILES are canonicalized."""
        # "OCC" and "CCO" are the same molecule
        mol1 = Molecule(molecule_id="TEST_007", smiles="OCC")
        result1 = self.tool._validate_molecule(mol1)
        
        mol2 = Molecule(molecule_id="TEST_008", smiles="CCO")
        result2 = self.tool._validate_molecule(mol2)
        
        assert result1['is_valid'] is True
        assert result2['is_valid'] is True
        # Both should have same canonical SMILES
        assert result1['smiles_canonical'] == result2['smiles_canonical']


# =============================================================================
# Integration Tests with TriageAgent
# =============================================================================

class TestAgentValidityIntegration:
    """Test ValidityTool integration with TriageAgent workflow."""
    
    def setup_method(self):
        """Initialize agent with ValidityTool before each test."""
        tools = [ValidityTool()]
        policy_engine = PolicyEngine()
        agent_logger = logging.getLogger("test_agent")
        
        self.agent = TriageAgent(
            tools=tools,
            policy_engine=policy_engine,
            logger=agent_logger
        )
    
    def test_valid_molecule_proceeds(self):
        """Test that valid molecules proceed through agent workflow."""
        state = self.agent.run(
            molecule_id="VALID_001",
            raw_input="CCO"  # Ethanol
        )
        
        # Should complete without early termination
        assert state is not None
        assert state.molecule_id == "VALID_001"
        
        # Check that ValidityTool ran
        assert 'ValidityTool' in state.tool_results
        validity_result = state.tool_results['ValidityTool']
        assert validity_result['is_valid'] is True
    
    def test_invalid_molecule_terminates_early(self):
        """Test that invalid molecules cause early termination."""
        state = self.agent.run(
            molecule_id="INVALID_001",
            raw_input="C(C)(C)(C)(C)C"  # Invalid valence
        )
        
        # Should still return state (not crash)
        assert state is not None
        
        # Check that molecule was marked invalid
        assert 'ValidityTool' in state.tool_results
        validity_result = state.tool_results['ValidityTool']
        assert validity_result['is_valid'] is False
        
        # Should have terminated early
        assert state.is_terminated() is True
    
    def test_empty_smiles_handled_gracefully(self):
        """Test that empty SMILES doesn't crash the system."""
        state = self.agent.run(
            molecule_id="EMPTY_001",
            raw_input=""
        )
        
        # Should not raise exception
        assert state is not None
        
        # Should have been caught and handled
        assert state.is_terminated() is True


# =============================================================================
# Standalone Utility Function Tests
# =============================================================================

class TestValidateSmilesFunction:
    """Test the standalone validate_smiles() helper function."""
    
    def test_validate_simple_valid(self):
        """Test validation of simple valid SMILES."""
        is_valid, error = validate_smiles("CCO")
        assert is_valid is True
        assert error is None
    
    def test_validate_invalid(self):
        """Test validation of invalid SMILES."""
        is_valid, error = validate_smiles("C(C)(C)(C)(C)C")
        assert is_valid is False
        assert error is not None
    
    def test_validate_empty(self):
        """Test validation of empty string."""
        is_valid, error = validate_smiles("")
        assert is_valid is False
        assert "empty" in error.lower()
    
    def test_validate_whitespace(self):
        """Test validation of whitespace string."""
        is_valid, error = validate_smiles("   ")
        assert is_valid is False
        assert "empty" in error.lower()


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        self.tool = ValidityTool()
    
    def test_single_atom_molecule(self):
        """Test single-atom molecule (valid but minimal)."""
        mol = Molecule(molecule_id="EDGE_001", smiles="C")
        result = self.tool._validate_molecule(mol)
        
        assert result['is_valid'] is True
        assert result['num_atoms'] == 1
    
    def test_large_molecule(self):
        """Test validation of large molecule."""
        # Cholesterol (27 carbons)
        smiles = "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C"
        mol = Molecule(molecule_id="EDGE_002", smiles=smiles)
        result = self.tool._validate_molecule(mol)
        
        assert result['is_valid'] is True
        assert result['num_atoms'] > 20
    
    def test_charged_molecule(self):
        """Test molecule with formal charges."""
        # Ammonium ion NH4+
        mol = Molecule(molecule_id="EDGE_003", smiles="[NH4+]")
        result = self.tool._validate_molecule(mol)
        
        # Should be valid (RDKit handles charges)
        assert result['is_valid'] is True
    
    def test_radical_molecule(self):
        """Test molecule with radical."""
        # Methyl radical
        mol = Molecule(molecule_id="EDGE_004", smiles="[CH3]")
        result = self.tool._validate_molecule(mol)
        
        # Should be valid (radicals are chemically meaningful)
        assert result['is_valid'] is True
    
    def test_multiple_components(self):
        """Test molecule with multiple disconnected components."""
        # Sodium chloride (Na+ and Cl- as separate ions)
        mol = Molecule(molecule_id="EDGE_005", smiles="[Na+].[Cl-]")
        result = self.tool._validate_molecule(mol)
        
        # Should be valid
        assert result['is_valid'] is True
    
    def test_complex_stereochemistry(self):
        """Test molecule with stereochemistry."""
        # L-alanine with stereochemistry
        mol = Molecule(molecule_id="EDGE_006", smiles="C[C@H](N)C(=O)O")
        result = self.tool._validate_molecule(mol)
        
        assert result['is_valid'] is True


# =============================================================================
# Tool Integration via AgentState
# =============================================================================

class TestToolStateIntegration:
    """Test ValidityTool's run() method with AgentState."""
    
    def test_run_with_string_input(self):
        """Test tool.run() with raw SMILES string."""
        tool = ValidityTool()
        state = AgentState(
            molecule_id="TEST_RUN_001",
            raw_input="CCO"
        )
        
        result = tool.run(state)
        
        assert result['is_valid'] is True
        assert result['molecule_id'] == "TEST_RUN_001"
        assert result['smiles_canonical'] == "CCO"
    
    def test_run_with_dict_input(self):
        """Test tool.run() with dictionary input."""
        tool = ValidityTool()
        state = AgentState(
            molecule_id="TEST_RUN_002",
            raw_input={
                'smiles': 'c1ccccc1',
                'name': 'Benzene',
                'metadata': {'source': 'test'}
            }
        )
        
        result = tool.run(state)
        
        assert result['is_valid'] is True
        assert result['molecule_id'] == "TEST_RUN_002"
    
    def test_run_with_molecule_input(self):
        """Test tool.run() with Molecule object."""
        tool = ValidityTool()
        mol = Molecule(molecule_id="TEST_RUN_003", smiles="CC(=O)O")
        state = AgentState(
            molecule_id="TEST_RUN_003",
            raw_input=mol
        )
        
        result = tool.run(state)
        
        assert result['is_valid'] is True
        assert result['smiles_canonical'] == "CC(=O)O"
    
    def test_run_with_invalid_input_type(self):
        """Test tool.run() with unsupported input type."""
        tool = ValidityTool()
        state = AgentState(
            molecule_id="TEST_RUN_004",
            raw_input=12345  # Invalid type
        )
        
        result = tool.run(state)
        
        assert result['is_valid'] is False
        assert 'Unknown input type' in result['error_message']


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])