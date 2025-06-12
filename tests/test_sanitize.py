import pytest
from rdkit import Chem
from peppr.sanitize import sanitize


@pytest.mark.parametrize("multiple_problems", [False, True])
@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccc2nc3c(c(=O)c2c1)cccc3",
        "n1cccc1",
        "CN(C)(C)C",
        "C1CCCC=O1",
    ],
)
def test_sanitize(smiles, multiple_problems):
    """
    Check if :func:`sanitize()` is able to fix the chemistry problems in the
    given molecules.

    Also check if molecules with the same problem appearing multiple times can
    be solved, by creating a :class:`Mol` containing two copies of the molecule.
    """
    if multiple_problems:
        smiles = smiles + "." + smiles
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    assert len(Chem.DetectChemistryProblems(mol)) > 0
    sanitize(mol)
    assert len(Chem.DetectChemistryProblems(mol)) == 0
