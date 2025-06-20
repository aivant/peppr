import itertools
from pathlib import Path
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest
from biotite.interface import rdkit as rdkit_interface
from rdkit import Chem
import peppr


@pytest.mark.parametrize(
    ["smiles", "ph", "ref_charges"],
    [
        # test some edge cases
        ("NC1CCCCC1", 4.0, [1, 0, 0, 0, 0, 0, 0]),
        ("Nc1ncccc1", 4.0, [1, 0, 0, 0, 0, 0, 0]),
        ("NC1=CCCCC1", 4.0, [1, 0, 0, 0, 0, 0, 0]),
        ("NC1=NCCCC1", 4.0, [0, 0, 1, 0, 0, 0, 0]),
        ("NC1CCCCC1", 7.4, [1, 0, 0, 0, 0, 0, 0]),
        ("Nc1ncccc1", 7.4, [0, 0, 0, 0, 0, 0, 0]),
        ("NC1=CCCCC1", 7.4, [0, 0, 0, 0, 0, 0, 0]),
        ("NC1=NCCCC1", 7.4, [0, 0, 1, 0, 0, 0, 0]),
    ],
)
def test_estimate_formal_charges_from_smiles(smiles, ph, ref_charges):
    """
    Check if formal charges are estimated correctly for molecules from SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    atoms = rdkit_interface.from_mol(mol, add_hydrogen=False)
    charges = peppr.estimate_formal_charges(atoms, ph)
    assert list(charges) == ref_charges


@pytest.mark.parametrize("include_hydrogen", [False, True])
@pytest.mark.parametrize(
    ["comp_name", "ph", "ref_charged_atoms"],
    [
        # Benzene: Does not get protonated/deprotonated
        ("BNZ", 7.0, {}),
        # Tryptophan: Ring amide should not get protonated
        ("TRP", 7.0, {"N": 1, "OXT": -1}),
        # Citrate: Multiple carboxy groups that match the same SMARTS pattern
        ("CIT", 7.0, {"O2": -1, "O4": -1, "O6": -1}),
        # Metal ions: Do not get protonated, but still have a charge
        ("MG", 0.0, {"MG": 2}),
        ("FE", 0.0, {"FE": 2}),
        # Alanine: Check pH sensitivity
        ("ALA", 0.0, {"N": 1}),
        ("ALA", 14.0, {"OXT": -1}),
        # Lysine: check pH sensitivity including uncharging
        ("LYS", 7.4, {"N": 1, "NZ": 1, "OXT": -1}),
        ("LYS", 0, {"N": 1, "NZ": 1}),
        ("LYS", 11, {"OXT": -1}),
        # Aspartate: check pH sensitivity including uncharging
        ("ASP", 7.4, {"N": 1, "OD2": -1, "OXT": -1}),
        ("ASP", 0, {"N": 1}),
        # Asparigine: do not charge amide nitrogens
        ("ASN", 7.4, {"N": 1, "OXT": -1}),
        # Cysteine
        ("CYS", 7.4, {"N": 1, "OXT": -1}),
        ("CYS", 9, {"N": 1, "SG": -1, "OXT": -1}),
        # Tyrosine
        ("TYR", 7.4, {"N": 1, "OXT": -1}),
        ("TYR", 12, {"OH": -1, "OXT": -1}),
        # Threonine
        ("THR", 7.4, {"N": 1, "OXT": -1}),
        ("THR", 12, {"OXT": -1}),
        ("THR", 17, {"OG1": -1, "OXT": -1}),
        # guanidine
        ("GAI", 7.4, {"N1": 1}),
        ("GAI", 14, {}),
        # phenol
        ("IPH", 7.4, {}),
        ("IPH", 14, {"O1": -1}),
        # ethanol
        ("EOH", 7.4, {}),
        ("EOH", 17, {"O": -1}),
        # acetic acid
        ("ACT", 7.4, {"OXT": -1}),
        ("ACT", 4, {}),
        # hydroxyamide example
        ("SHH", 7.4, {}),
        ("SHH", 10, {"O1": -1}),
        # ascorbic acid (vinylogous carboxylic acid)
        ("ASC", 7.4, {"O3": -1}),
        ("ASC", 4, {}),
        # imine example
        ("MFG", 7.4, {}),
        ("MFG", 3, {"N10": 1}),
        # tetrazole and imidazole motifs
        ("3GK", 7.4, {"N1": -1}),
        ("3GK", 5, {"N1": -1, "N21": 1}),
        ("3GK", 4, {"N21": 1}),
        # thiophenol example
        ("JKE", 7.4, {"SD": -1, "OD2": -1}),
        ("JKE", 6, {"OD2": -1}),
        ("JKE", 4, {}),
        # sulfinic acid example
        ("OBP", 7.4, {"OX2": -1}),
        # sulfonic acid example
        ("EPE", 7.4, {"N1": 1, "N4": 1, "O3S": -1}),
        # sulfate ion, treated the same as sulfonic acid
        ("SO4", 7.3, {"O3": -1, "O4": -1}),
        # ATP
        ("ATP", 7.4, {"O2G": -1, "O3G": -1, "O2B": -1, "O2A": -1}),
        # phosphonic acid example
        ("BOY", 7.4, {"OAB": -1, "OAD": -1}),
        # diphosphate - this may be overcharged at pH ~7 and below
        ("POP", 7.4, {"O2": -1, "O3": -1, "O5": -1, "O6": -1}),
    ],
)
def test_estimate_formal_charges(comp_name, ph, ref_charged_atoms, include_hydrogen):
    """
    Check if formal charges are estimated correctly for known small molecules.
    """
    atoms = info.residue(comp_name)
    # Do not rely on charges extracted from the CCD
    atoms.del_annotation("charge")
    if not include_hydrogen:
        atoms = atoms[atoms.element != "H"]

    charges = peppr.estimate_formal_charges(atoms, ph)
    test_charged_atoms = {
        atom_name: charge
        for atom_name, charge in zip(atoms.atom_name, charges)
        if charge != 0
    }

    assert len(charges) == atoms.array_length()
    assert test_charged_atoms == ref_charged_atoms


def test_estimate_formal_charges_peptide():
    """
    Check if a peptide has the expected charges at the termini and the acidic/basic
    residues.
    """
    pdbx_file = pdbx.CIFFile.read(Path(__file__).parent / "data" / "pdb" / "1a3n.cif")
    atoms = pdbx.get_structure(
        pdbx_file,
        model=1,
        include_bonds=True,
        use_author_fields=False,
    )
    atoms = atoms[atoms.chain_id == "A"]

    charges = peppr.estimate_formal_charges(atoms)

    # N-terminus
    assert np.all(
        charges[(atoms.res_id == atoms.res_id[0]) & (atoms.atom_name == "N")] == 1
    )
    # C-terminus
    assert np.all(
        charges[(atoms.res_id == atoms.res_id[-1]) & (atoms.atom_name == "OXT")] == -1
    )
    # Acidic/basic residues
    for start, stop in itertools.pairwise(
        # Omit the termini
        struc.get_residue_starts(atoms, add_exclusive_stop=True)[1:-1]
    ):
        if atoms.res_name[start] in ("ASP", "GLU"):
            expected_charge = -1
        elif atoms.res_name[start] in ("LYS", "ARG"):
            expected_charge = 1
        else:
            expected_charge = 0
        assert np.sum(charges[start:stop]) == expected_charge
