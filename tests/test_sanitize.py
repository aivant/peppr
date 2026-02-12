import biotite.interface.rdkit as rdkit_interface
import biotite.structure.info as info
import pytest
from rdkit import Chem
from peppr.sanitize import sanitize


@pytest.mark.parametrize("multiple_problems", [False, True])
@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccc2nc3c(c(=O)c2c1)cccc3",
        "O=c1c(cccc2)c2c(Cc3ccccc3)nn1",
        "n1cccc1",
        "CN(C)(C)C",
        "C1CCCC=O1",
        "c1cncn1",
    ],
)
def test_sanitize_from_smiles(smiles, multiple_problems):
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


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("has_charge", [False, True])
@pytest.mark.parametrize("multiple_problems", [False, True])
@pytest.mark.parametrize(
    "comp_name",
    [
        "TRP",
        "HIS",
        "ACH",
    ],
)
def test_sanitize_from_atom_array(comp_name, multiple_problems, has_charge, seed):
    """
    Same as :func:`test_sanitize_from_smiles()`, but the molecule originates
    from an :class:`AtomArray`.
    Furthermore, the atom order is randomized to ensure that correct sanitization
    does not depend on the order of atoms.
    """
    atoms = info.residue(comp_name)
    atoms = atoms[atoms.element != "H"]
    if not has_charge:
        # Do not rely on preset charges
        atoms.charge[:] = 0
    if multiple_problems:
        atoms = atoms + atoms
    # rng = np.random.default_rng(seed)
    # atoms = atoms[rng.permutation(atoms.array_length())]

    mol = rdkit_interface.to_mol(atoms, explicit_hydrogen=False)
    sanitize(mol)
    assert len(Chem.DetectChemistryProblems(mol)) == 0
