import pytest
from rdkit import Chem
from peppr.sanitize import sanitize


@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccc2nc3c(c(=O)c2c1)cccc3",
        "n1cccc1",
        "CN(C)(C)C",
        "C1CCCC=O1",
    ],
)
def test_soft_rdkit_sanitize_mol(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    assert len(Chem.DetectChemistryProblems(mol)) > 0
    sanitize(mol)
    assert len(Chem.DetectChemistryProblems(mol)) == 0
