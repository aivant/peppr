import biotite.interface.rdkit as rdkit_interface
import pytest
from rdkit import Chem
from peppr.metric_reffree import (
    LigandValenceViolations,
    _count_valence_violations,
)


def _test_mols_valence_violations() -> dict[Chem.Mol, int]:
    good_smiles = [
        "C1=CC=CC=C1",
        "C1=CC=CC=C1[H]",
        "N(=O)=O",
        "C=P(=O)O",
        "O=Cl(=O)O",
        "OC1C(COP(=O)(OP(=O)(OP(=O)(O)O)O)O)OC(C1O)n1cnc2c1ncnc2N",
        "CO(C)C",  # Oxygen charge +1
    ]

    one_viol = [
        "C[C@H](FC)C",
        "CF(C)C",
    ]

    two_viols = ["NC(O)NCCC1:C:C:C:C([C@H](C=F)/C=O/C(=O)(O)O):C:1"]

    params = Chem.SmilesParserParams()
    params.sanitize = False
    good_mols = {Chem.MolFromSmiles(smile, params=params): 0 for smile in good_smiles}
    one_viol = {Chem.MolFromSmiles(smile, params=params): 1 for smile in one_viol}
    two_viols = {Chem.MolFromSmiles(smile, params=params): 2 for smile in two_viols}
    return good_mols | one_viol | two_viols



@pytest.mark.parametrize(
    ["mol", "expected_viols"],
    [(mol, num_exp_viols) for mol, num_exp_viols in _test_mols_valence_violations().items()],
)
def test_count_valence_violations(mol, expected_viols):
    """
    Check if the correct number of valence violations are found by
    :func:`_count_valence_violations` by checking the metric against
    reference molecules with a known number of violations.
    """
    smiles = Chem.MolToSmiles(mol)
    aarray = rdkit_interface.from_mol(mol, add_hydrogen=False)
    num_violations = _count_valence_violations(aarray)
    assert num_violations == expected_viols, (
        f"Expected {expected_viols} violations for {smiles}, got {num_violations}"
    )


@pytest.mark.parametrize(
    ["mol", "expected_viols"],
    [(mol, num_exp_viols) for mol, num_exp_viols in _test_mols_valence_violations().items()],
)
def test_ligand_valence_violations(mol, expected_viols):
    """
    Check if the correct number of valence violations are found by
    :class:`LigandValenceViolations` by checking the metric against
    reference molecules with a known number of violations.
    """
    metric = LigandValenceViolations()
    smiles = Chem.MolToSmiles(mol)
    atom_stack = rdkit_interface.from_mol(mol, add_hydrogen=False)
    aarray = atom_stack[0]
    aarray.hetero[:] = True
    num_violations = metric.evaluate(aarray)
    assert num_violations == expected_viols, (
        f"Expected {expected_viols} violations for {smiles}, got {num_violations}"
    )
