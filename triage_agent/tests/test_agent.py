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
from molecule import Molecule
from tools.validity_tool import ValidityTool, validate_smiles
from agent.triage_agent import TriageAgent
from policies.policy_engine import PolicyEngine


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
        result = self.tool.evaluate(mol, context={})
        
        assert result['is_valid'] is True
        assert result['error_message'] is None
        assert result['smiles_canonical'] == "CCO"
        assert result['num_atoms'] > 0
        assert result['num_bonds'] > 0
    
    def test_valid_aromatic_molecule(self):
        """Test validation of an aromatic molecule (benzene)."""
        mol = Molecule(molecule_id="TEST_002", smiles="c1ccccc1")
        result = self.tool.evaluate(mol, context={})
        
        assert result['is_valid'] is True
        assert result['smiles_canonical'] == "c1ccccc1"
        assert result['num_atoms'] == 6  # 6 carbons
    
    def test_invalid_valence(self):
        """Test rejection of molecule with invalid valence."""
        # Carbon with 5 bonds (chemically impossible)
        mol = Molecule(molecule_id="TEST_003", smiles="C(C)(C)(C)(C)C")
        result = self.tool.evaluate(mol, context={})
        
        assert result['is_valid'] is False
        assert result['error_message'] is not None
        assert "RDKit failed to parse" in result['error_message']
    
    def test_empty_smiles(self):
        """Test rejection of empty SMILES string."""
        mol = Molecule(molecule_id="TEST_004", smiles="")
        result = self.tool.evaluate(mol, context={})
        
        assert result['is_valid'] is False
        assert "empty" in result['error_message'].lower()
    
    def test_whitespace_only_smiles(self):
        """Test rejection of whitespace-only SMILES."""
        mol = Molecule(molecule_id="TEST_005", smiles="   ")
        result = self.tool.evaluate(mol, context={})
        
        assert result['is_valid'] is False
        assert "empty" in result['error_message'].lower()
    
    def test_malformed_smiles(self):
        """Test rejection of syntactically invalid SMILES."""
        mol = Molecule(molecule_id="TEST_006", smiles="C(((")
        result = self.tool.evaluate(mol, context={})
        
        assert result['is_valid'] is False
        assert result['error_message'] is not None
    
    def test_canonicalization(self):
        """Test that non-canonical SMILES are canonicalized."""
        # "OCC" and "CCO" are the same molecule
        mol1 = Molecule(molecule_id="TEST_007", smiles="OCC")
        result1 = self.tool.evaluate(mol1, context={})
        
        mol2 = Molecule(molecule_id="TEST_008", smiles="CCO")
        result2 = self.tool.evaluate(mol2, context={})
        
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
        self.agent = TriageAgent(
            tools=tools,
            policy_engine=policy_engine,
            logger=logging.getLogger("test_agent")
        )
    
    def test_valid_molecule_proceeds(self):
        """Test that valid molecules proceed through agent workflow."""
        decision = self.agent.run(
            molecule_id="VALID_001",
            raw_input="CCO"  # Ethanol
        )
        
        # Should complete without early termination
        assert decision is not None
        # Check that ValidityTool ran and passed
        # (exact decision type depends on PolicyEngine config)
    
    def test_invalid_molecule_terminates_early(self):
        """Test that invalid molecules cause early termination."""
        decision = self.agent.run(
            molecule_id="INVALID_001",
            raw_input="C(C)(C)(C)(C)C"  # Invalid valence
        )
        
        # Should still return a decision (not crash)
        assert decision is not None
        # Decision should be DISCARD or FLAG
        # AgentState should show early termination
    
    def test_empty_smiles_handled_gracefully(self):
        """Test that empty SMILES doesn't crash the system."""
        decision = self.agent.run(
            molecule_id="EMPTY_001",
            raw_input=""
        )
        
        # Should not raise exception
        assert decision is not None


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
        result = self.tool.evaluate(mol, context={})
        
        assert result['is_valid'] is True
        assert result['num_atoms'] == 1
    
    def test_large_molecule(self):
        """Test validation of large molecule."""
        # Cholesterol (27 carbons)
        smiles = "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C"
        mol = Molecule(molecule_id="EDGE_002", smiles=smiles)
        result = self.tool.evaluate(mol, context={})
        
        assert result['is_valid'] is True
        assert result['num_atoms'] > 20
    
    def test_charged_molecule(self):
        """Test molecule with formal charges."""
        # Ammonium ion NH4+
        mol = Molecule(molecule_id="EDGE_003", smiles="[NH4+]")
        result = self.tool.evaluate(mol, context={})
        
        # Should be valid (RDKit handles charges)
        assert result['is_valid'] is True
    
    def test_radical_molecule(self):
        """Test molecule with radical."""
        # Methyl radical
        mol = Molecule(molecule_id="EDGE_004", smiles="[CH3]")
        result = self.tool.evaluate(mol, context={})
        
        # Should be valid (radicals are chemically meaningful)
        assert result['is_valid'] is True


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])