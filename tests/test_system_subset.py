import biotite.structure as struc
import biotite.structure.info as info
import numpy as np
import pytest
from peppr.system_subset import IsolatedLigandSelector


def _dummy_protein_ligand() -> tuple[struc.AtomArray, struc.AtomArray]:
    """Assemble a test system with a protein chain and a ligand."""
    # Create a simple protein chain (3 alanines)
    protein: struc.AtomArray = info.residue("ALA")
    protein.chain_id[:] = "P"
    protein.add_annotation("mask", bool)
    protein = struc.concatenate([protein] * 3)

    # Create a small molecule ligand
    ligand: struc.AtomArray = info.residue("GLY")
    ligand.chain_id[:] = "L"
    ligand.hetero[:] = True
    ligand.add_annotation("mask", bool)
    ligand.mask[:] = True

    # Combine into a single AtomArray
    return protein, ligand


def _assemble_complex() -> tuple[struc.AtomArray]:
    """Assemble a test system with a protein chain and a ligand."""
    protein, ligand = _dummy_protein_ligand()
    return struc.concatenate([protein, ligand])


def _assemble_multi_ligand_system():
    """Assemble a system with a protein and multiple isolated ligands."""
    protein, ligand1 = _dummy_protein_ligand()
    ligand2 = ligand1.copy()
    ligand2.chain_id[:] = "M"

    # Combine all parts
    multi_ligand_system = struc.concatenate([protein, ligand1, ligand2])
    return multi_ligand_system


def _assemble_protein_only_system() -> tuple[struc.AtomArray]:
    """Assemble a system with only protein chains."""
    protein1 = info.residue("ALA")
    protein1.chain_id[:] = "A"
    protein1.add_annotation("mask", bool)
    protein2 = info.residue("GLY")
    protein2.chain_id[:] = "B"
    protein2.add_annotation("mask", bool)
    return struc.concatenate([protein1, protein2])


def _assemble_ligand_only_system():
    """Assemble a system with only ligand chains."""
    ligand1 = info.residue("ALA")
    ligand1.chain_id[:] = "C"
    ligand1.hetero[:] = True
    ligand1.add_annotation("mask", bool)
    ligand1.mask[:] = True
    ligand2 = info.residue("GLY")
    ligand2.chain_id[:] = "D"
    ligand2.hetero[:] = True
    ligand2.add_annotation("mask", bool)
    ligand2.mask[:] = True
    return struc.concatenate([ligand1, ligand2])


def _get_masks(system: struc.AtomArray) -> list[np.ndarray]:
    chain_starts = struc.get_chain_starts(system)
    chain_masks = struc.get_chain_masks(system, chain_starts)
    ligand_masks = chain_masks[(chain_masks & system.mask).any(axis=-1)]
    return ligand_masks


@pytest.mark.parametrize(
    "system",
    [
        _assemble_complex(),
        _assemble_multi_ligand_system(),
        _assemble_protein_only_system(),
        _assemble_ligand_only_system(),
    ],
    ids=[
        "protein_ligand_complex",
        "protein_multi_ligand",
        "protein_only",
        "ligand_only",
    ],
)
def test_isolated_ligand_selector(system):
    """
    Test IsolatedLigandSelector returns correct masks for various system types.
    """
    expected_masks = _get_masks(system)
    selector = IsolatedLigandSelector()

    masks = selector.select(system)

    assert np.equal(masks, expected_masks).all()


def test_isolated_ligand_selector_empty_input():
    """
    Test IsolatedLigandSelector handles an empty AtomArray.
    """
    selector = IsolatedLigandSelector()
    empty_array = struc.AtomArray(0)
    masks = selector.select(empty_array)
    assert len(masks) == 0
    assert isinstance(masks, np.ndarray)
