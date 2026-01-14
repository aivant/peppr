import functools
from pathlib import Path
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest
from biotite.interface import rdkit as rdkit_interface
import peppr
from peppr.common import ACCEPTOR_PATTERN, DONOR_PATTERN, HBOND_DISTANCE_SCALING


def test_find_atoms_by_pattern():
    """
    Check if for a selected structure the manually identified hydrogen bond
    donors/acceptors are found by :func:`find_atoms_by_pattern()`.

    As the manual identification may miss some donors/acceptors, expect the
    results from :func:`find_atoms_by_pattern()` to be a superset of the
    expected results.
    """
    LIGAND_DONORS = [(300, "N1"), (300, "N2")]
    LIGAND_ACCEPTORS = [(300, "O11"), (300, "S1"), (300, "O3")]
    RECEPTOR_DONORS = [(49, "N"), (90, "OG1"), (23, "ND2"), (43, "OH")]
    RECEPTOR_ACCEPTORS = [(45, "OG"), (128, "OD2")]

    pdbx_file = pdbx.CIFFile.read(Path(__file__).parent / "data" / "pdb" / "2rtg.cif")
    atoms = pdbx.get_structure(
        pdbx_file,
        model=1,
        include_bonds=True,
    )
    # Remove salt and water
    atoms = atoms[~struc.filter_solvent(atoms) & ~struc.filter_monoatomic_ions(atoms)]
    # Focus on a single monomer
    atoms = atoms[atoms.chain_id == "B"]

    # Use only heavy atoms
    atoms = atoms[atoms.element != "H"]
    receptor_mask = struc.filter_amino_acids(atoms)
    ligand_mask = atoms.res_name == "BTN"
    contact_measurement = peppr.ContactMeasurement(
        atoms[receptor_mask], atoms[ligand_mask], 10.0
    )

    ligand = contact_measurement._ligand
    pocket = contact_measurement._binding_site
    ligand_mol = contact_measurement._ligand_mol
    pocket_mol = contact_measurement._binding_site_mol

    for structure, mol, pattern, ref_atoms in [
        (ligand, ligand_mol, DONOR_PATTERN, LIGAND_DONORS),
        (ligand, ligand_mol, ACCEPTOR_PATTERN, LIGAND_ACCEPTORS),
        (pocket, pocket_mol, DONOR_PATTERN, RECEPTOR_DONORS),
        (pocket, pocket_mol, ACCEPTOR_PATTERN, RECEPTOR_ACCEPTORS),
    ]:
        indices = peppr.find_atoms_by_pattern(mol, pattern)
        test_atoms = set(
            [(structure.res_id[i], structure.atom_name[i]) for i in indices]
        )
        ref_atoms = set(ref_atoms)
        assert test_atoms.issuperset(ref_atoms), ref_atoms.difference(test_atoms)


@pytest.mark.parametrize("ideal_angle", [None, np.deg2rad(180)])
def test_hydrogen_bond_identification(ideal_angle):
    """
    Check for a PLI complex (biotin-streptavidin) whether hydrogen bonds,
    measured using explicit hydrogen atoms, are correctly recovered with heavy atoms
    only using :meth:`ContactMeasurement.find_contacts_by_pattern()`.
    """
    FALSE_POSITIVE_FACTOR = 2.0

    pdbx_file = pdbx.CIFFile.read(Path(__file__).parent / "data" / "pdb" / "2rtg.cif")
    atoms = pdbx.get_structure(
        pdbx_file,
        model=1,
        include_bonds=True,
    )
    # Remove salt and water
    atoms = atoms[~struc.filter_solvent(atoms) & ~struc.filter_monoatomic_ions(atoms)]
    # Focus on a single monomer
    atoms = atoms[atoms.chain_id == "B"]

    receptor_mask = struc.filter_amino_acids(atoms)
    ligand_mask = atoms.res_name == "BTN"
    ref_contact_indices = struc.hbond(atoms, receptor_mask, ligand_mask)
    # Keep the residue ID and atom name for easier comparison to test values
    # and better visual inspection
    ref_contacts = []
    for donor_i, _, acceptor_i in ref_contact_indices:
        ref_contacts.append(
            (
                atoms.res_id[donor_i],
                atoms.atom_name[donor_i],
                atoms.res_id[acceptor_i],
                atoms.atom_name[acceptor_i],
            )
        )

    # Use only heavy atoms
    atoms = atoms[atoms.element != "H"]
    receptor_mask = struc.filter_amino_acids(atoms)
    ligand_mask = atoms.res_name == "BTN"
    contact_measurement = peppr.ContactMeasurement(
        atoms[receptor_mask], atoms[ligand_mask], 10.0
    )
    # If a 'straight' hydrogen bond is assumed as ideal, increase the tolerance,
    # as hydrogen bonds are typically not perfectly straight
    tolerance = np.deg2rad(30) if ideal_angle is None else np.deg2rad(90)
    test_contacts = []
    # Measure cases where the receptor is the donor and the ligand the acceptor
    for receptor_i, ligand_i in contact_measurement.find_contacts_by_pattern(
        DONOR_PATTERN,
        ACCEPTOR_PATTERN,
        HBOND_DISTANCE_SCALING,
        ideal_angle,
        ideal_angle,
        tolerance,
    ):
        test_contacts.append(
            (
                atoms.res_id[receptor_mask][receptor_i],
                atoms.atom_name[receptor_mask][receptor_i],
                atoms.res_id[ligand_mask][ligand_i],
                atoms.atom_name[ligand_mask][ligand_i],
            )
        )
    # Measure cases where the receptor is the acceptor and the ligand the donor
    for receptor_i, ligand_i in contact_measurement.find_contacts_by_pattern(
        ACCEPTOR_PATTERN,
        DONOR_PATTERN,
        HBOND_DISTANCE_SCALING,
        ideal_angle,
        ideal_angle,
        tolerance,
    ):
        test_contacts.append(
            (
                atoms.res_id[ligand_mask][ligand_i],
                atoms.atom_name[ligand_mask][ligand_i],
                atoms.res_id[receptor_mask][receptor_i],
                atoms.atom_name[receptor_mask][receptor_i],
            )
        )
    assert set(ref_contacts).issubset(test_contacts), set(ref_contacts).difference(
        test_contacts
    )
    # Although the hydrogen bonds measured only based on heavy atoms may contain false
    # positives, there should not be excessively many false positives
    assert len(test_contacts) <= len(ref_contacts) * FALSE_POSITIVE_FACTOR


@pytest.mark.parametrize("use_resonance", [False, True])
@pytest.mark.parametrize(
    ["pdb_id", "ligand_res_name", "expected_contacts"],
    [
        (
            "1a3n",
            "HEM",
            [
                # receptor res_id, possible receptor atom_names, possible ligand atom_names
                (45, ["NE2"], ["O1D", "O2D"]),
                (61, ["NZ"], ["O1A", "O2A"]),
            ],
        ),
        (
            "3eca",
            "ASP",
            [
                (90, ["OD1", "OD2"], ["N"]),
            ],
        ),
    ],
)
def test_salt_bridge_identification(
    pdb_id, ligand_res_name, expected_contacts, use_resonance
):
    """
    Check for the given PLI complexes whether the expected salt bridges are found.
    """
    pdbx_file = pdbx.CIFFile.read(
        Path(__file__).parent / "data" / "pdb" / f"{pdb_id}.cif"
    )
    atoms = pdbx.get_structure(
        pdbx_file,
        model=1,
        include_bonds=True,
    )
    # Remove salt, water and hydrogen
    atoms = peppr.standardize(atoms)
    # Focus on a single monomer
    atoms = atoms[atoms.chain_id == struc.get_chains(atoms)[0]]
    receptor = atoms[struc.filter_amino_acids(atoms) & ~atoms.hetero]
    ligand = atoms[(atoms.res_name == ligand_res_name) & atoms.hetero]

    # These are 'possible contacts', as a bridge may be formed only from any one of the
    # atom names to one of the other atom names
    possible_ref_contacts = set()
    for receptor_res_id, receptor_atom_names, ligand_atom_names in expected_contacts:
        for receptor_atom_name in receptor_atom_names:
            for ligand_atom_name in ligand_atom_names:
                possible_ref_contacts.add(
                    (
                        receptor_res_id,
                        receptor_atom_name,
                        ligand_atom_name,
                    )
                )

    # Use a slightly lower pH value to force protonation of the histidine side chain
    contact_measurement = peppr.ContactMeasurement(receptor, ligand, cutoff=4.0, ph=6.0)
    test_contacts = contact_measurement.find_salt_bridges(use_resonance=use_resonance)

    assert len(test_contacts) == len(expected_contacts)
    for i, j in test_contacts:
        contact = (
            receptor.res_id[i].item(),
            receptor.atom_name[i].item(),
            ligand.atom_name[j].item(),
        )
        assert contact in possible_ref_contacts


def test_find_stacking_interactions():
    """Test π-stacking interaction detection between protein and ligand.
    It uses pdb 1acj known to have stacking interactions between the protein and ligand."""

    pdbx_file = pdbx.CIFFile.read(Path(__file__).parent / "data" / "pdb" / "1acj.cif")
    atoms = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    receptor = atoms[~atoms.hetero]
    ligand = atoms[atoms.hetero]

    contact_measurement = peppr.ContactMeasurement(receptor, ligand, 10.0)
    interactions = contact_measurement.find_stacking_interactions()
    assert len(interactions) > 0

    # assert interactions are between rings of Tryptophan and Phenylalanine residues with ligand THA
    valid_residues = {"TRP", "PHE"}
    for protein_indices, ligand_indices, _ in interactions:
        assert all(
            atom.res_name in valid_residues for atom in receptor[protein_indices]
        ), "Found interaction with unexpected residue"
        assert all(atom.res_name == "THA" for atom in ligand[ligand_indices]), (
            "Found interaction with unexpected ligand"
        )


@pytest.mark.parametrize(
    "contact_method",
    [
        functools.partial(
            peppr.ContactMeasurement.find_contacts_by_pattern,
            receptor_pattern="*",
            ligand_pattern="*",
            distance_scaling=(0.0, 1.0),
        ),
        peppr.ContactMeasurement.find_salt_bridges,
    ],
    ids=["find_contacts_by_pattern", "find_salt_bridges"],
)
def test_no_contacts(contact_method):
    """
    Check if the output array shape is still correct if no contacts are found.
    To do this, place the receptor and ligand far apart from each other.
    """
    ligand = info.residue("ALA")
    ligand = ligand[ligand.element != "H"]
    receptor = ligand.copy()
    # Move the receptor far away from the ligand
    receptor.coord += 1000

    contact_measurement = peppr.ContactMeasurement(receptor, ligand, 10.0)
    contacts = contact_method(contact_measurement)
    assert contacts.shape == (0, 2)


@pytest.mark.parametrize(
    "contact_method",
    [
        functools.partial(
            peppr.ContactMeasurement.find_contacts_by_pattern,
            receptor_pattern="*",
            ligand_pattern="*",
            distance_scaling=(0.0, 1.0),
        ),
        peppr.ContactMeasurement.find_salt_bridges,
    ],
    ids=["find_contacts_by_pattern", "find_salt_bridges"],
)
def test_no_bonds(contact_method):
    """
    Check if :class:`ContactMeasurement` still works if the receptor and ligand
    do not have any bonds.
    """
    ligand = info.residue("ALA")
    ligand = ligand[ligand.element != "H"]
    # Remove all bonds
    ligand.bonds = struc.BondList(ligand.array_length())
    receptor = ligand.copy()
    receptor.coord[:, 0] += 5

    contact_measurement = peppr.ContactMeasurement(receptor, ligand, 10.0)
    contact_method(contact_measurement)


def test_find_charged_atoms_in_resonance_structures():
    """
    Test finding charged atoms in resonance structures using a real ligand from a PDB file.
    """
    pdbx_file = pdbx.CIFFile.read(Path(__file__).parent / "data" / "pdb" / "3eca.cif")
    structure = pdbx.get_structure(
        pdbx_file,
        model=1,
        include_bonds=True,
    )
    structure = peppr.standardize(structure)
    structure = structure[structure.chain_id == "A"]
    ligand = structure[structure.hetero]

    # set annotations for the benefit of finding charged atoms
    ligand.set_annotation("charge", peppr.estimate_formal_charges(ligand, 7.4))
    # create rdkit ligand object
    ligand_mol = rdkit_interface.to_mol(ligand)
    try:
        peppr.sanitize(ligand_mol)
    except Exception:
        return np.nan
    ligand_charged_atoms = np.where(ligand.charge != 0)[0]
    assert np.equal(ligand_charged_atoms, [0, 7, 8]).all()

    # get charged atoms and their resonance groups
    pos_mask, neg_mask, ligand_conjugated_groups = (
        peppr.find_charged_atoms_in_resonance_structures(ligand_mol)
    )
    assert len(set(ligand_conjugated_groups)) < len(ligand_conjugated_groups), (
        "Some atoms are in the same conjugated group"
    )
    charged_atom_mask = pos_mask | neg_mask
    ligand_charged_in_resonance_atoms = np.where(charged_atom_mask)[0]
    assert set(ligand_charged_atoms) != set(ligand_charged_in_resonance_atoms), (
        "Charged atoms do not match those found in resonance structures"
    )
    assert np.equal(ligand_charged_in_resonance_atoms, [0, 3, 6, 7, 8]).all()
