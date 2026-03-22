# tools/descriptor_tool.py
"""
DescriptorTool — Week 9

Computes seven physicochemical properties from a molecule in a single RDKit pass:
    mol_weight, logp, tpsa, hbd, hba, rotatable_bonds, scaffold_smiles

Runs immediately after ValidityTool (position 2 in the pipeline), before SAScoreTool,
so scaffold_smiles is available early for diversity analysis.

Non-terminal: descriptor failures add a WARNING flag but never DISCARD or stop
the pipeline. All seven values default to None on failure so SQLite columns remain
nullable rather than containing bad data.
"""

import logging
import time
from typing import Any, Dict, Optional

from tools.base_tool import Tool
from agent.agent_state import AgentState

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class DescriptorTool(Tool):
    """
    Compute Lipinski/drug-likeness descriptors and Bemis-Murcko scaffold SMILES.

    Reads the mol object (or canonical SMILES) produced by ValidityTool.
    Stores results in state.tool_results['DescriptorTool'].
    """

    def __init__(self):
        self.name = "DescriptorTool"
        self.description = (
            "Computes MW, logP, TPSA, HBD, HBA, rotatable bonds, "
            "and Bemis-Murcko scaffold SMILES using RDKit."
        )
        self.logger = logging.getLogger(__name__)

        if not RDKIT_AVAILABLE:
            self.logger.error(
                "RDKit not available — DescriptorTool will return None for all descriptors."
            )

    def run(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute descriptor calculation.

        Args:
            state: AgentState after ValidityTool has run.

        Returns:
            Dict with keys: mol_weight, logp, tpsa, hbd, hba,
            rotatable_bonds, scaffold_smiles, error_message.
            Stored in state.tool_results['DescriptorTool'].
        """
        t0 = time.perf_counter()

        null_result = {
            "tool_name": self.name,
            "molecule_id": state.molecule_id,
            "mol_weight": None,
            "logp": None,
            "tpsa": None,
            "hbd": None,
            "hba": None,
            "rotatable_bonds": None,
            "scaffold_smiles": None,
            "error_message": None,
            "execution_time_ms": 0.0,
        }

        if not RDKIT_AVAILABLE:
            null_result["error_message"] = "RDKit not available"
            return null_result

        # --- Guard: validity must have passed ---
        validity = state.tool_results.get("ValidityTool", {})
        if validity and not validity.get("is_valid", True):
            null_result["error_message"] = "Molecule invalid — descriptors skipped"
            return null_result

        # --- Extract SMILES ---
        smiles = None
        if validity:
            smiles = validity.get("smiles_canonical") or validity.get("smiles")
        if not smiles:
            smiles = state.raw_input if isinstance(state.raw_input, str) else None

        if not smiles:
            null_result["error_message"] = "No SMILES available"
            return null_result

        # --- Build mol ---
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            null_result["error_message"] = f"RDKit could not parse: {smiles!r}"
            return null_result

        # --- Compute descriptors ---
        try:
            mol_weight = round(Descriptors.MolWt(mol), 3)
            logp = round(Descriptors.MolLogP(mol), 3)
            tpsa = round(Descriptors.TPSA(mol), 3)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        except Exception as exc:
            null_result["error_message"] = f"Descriptor calculation failed: {exc}"
            state.add_flag(
                reason=f"DescriptorTool failed: {exc}",
                source=self.name,
            )
            return null_result

        # --- Bemis-Murcko scaffold ---
        scaffold_smiles: Optional[str] = None
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is not None:
                scaffold_smiles = Chem.MolToSmiles(scaffold)
        except Exception as exc:
            self.logger.warning("Scaffold extraction failed for %s: %s", smiles, exc)
            # Non-fatal — scaffold_smiles stays None

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "tool_name": self.name,
            "molecule_id": state.molecule_id,
            "mol_weight": mol_weight,
            "logp": logp,
            "tpsa": tpsa,
            "hbd": hbd,
            "hba": hba,
            "rotatable_bonds": rotatable_bonds,
            "scaffold_smiles": scaffold_smiles,
            "error_message": None,
            "execution_time_ms": round(elapsed_ms, 2),
        }