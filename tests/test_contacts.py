import functools
from pathlib import Path
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest
from biotite.interface import rdkit as rdkit_interface
from rdkit import Chem
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
        atoms[receptor_mask], atoms[ligand_mask]
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


@pytest.mark.parametrize("use_tauts", [True, False])
@pytest.mark.parametrize("ideal_angle", [None, np.deg2rad(180)])
def test_hydrogen_bond_identification(ideal_angle, use_tauts):
    """
    Check for a PLI complex (biotin-streptavidin) whether hydrogen bonds,
    measured using explicit hydrogen atoms, are correctly recovered with heavy atoms
    only using :meth:`ContactMeasurement.find_contacts_by_pattern()`.
    """
    FALSE_POSITIVE_FACTOR = 1.5

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
    # Each reference H-bond is stored as
    # (lig_is_donor, lig_res_id, lig_atom_name, rec_res_id, rec_atom_name)
    ref_contacts = []
    for donor_i, _, acceptor_i in ref_contact_indices:
        lig_is_donor = ligand_mask[donor_i]
        donor_atom = (atoms.res_id[donor_i], atoms.atom_name[donor_i])
        acceptor_atom = (atoms.res_id[acceptor_i], atoms.atom_name[acceptor_i])
        lig_atom = donor_atom if lig_is_donor else acceptor_atom
        rec_atom = acceptor_atom if lig_is_donor else donor_atom
        ref_contacts.append(
            (lig_is_donor, lig_atom[0], lig_atom[1], rec_atom[0], rec_atom[1])
        )

    # Use only heavy atoms
    atoms = atoms[atoms.element != "H"]
    receptor_mask = struc.filter_amino_acids(atoms)
    ligand_mask = atoms.res_name == "BTN"
    contact_measurement = peppr.ContactMeasurement(
        atoms[receptor_mask],
        atoms[ligand_mask],
        use_tautomers=use_tauts,
    )
    # If a 'straight' hydrogen bond is assumed as ideal, increase the tolerance,
    # as hydrogen bonds are typically not perfectly straight
    tolerance = np.deg2rad(30) if ideal_angle is None else np.deg2rad(90)
    test_contacts = []
    for receptor_pattern, ligand_pattern in [
        (DONOR_PATTERN, ACCEPTOR_PATTERN),
        (ACCEPTOR_PATTERN, DONOR_PATTERN),
    ]:
        for receptor_i, ligand_i in contact_measurement.find_contacts_by_pattern(
            receptor_pattern,
            ligand_pattern,
            HBOND_DISTANCE_SCALING,
            ideal_angle,
            ideal_angle,
            tolerance,
        ):
            lig_is_donor = ligand_pattern == DONOR_PATTERN
            test_contacts.append(
                (
                    lig_is_donor,
                    atoms.res_id[ligand_mask][ligand_i],
                    atoms.atom_name[ligand_mask][ligand_i],
                    atoms.res_id[receptor_mask][receptor_i],
                    atoms.atom_name[receptor_mask][receptor_i],
                )
            )
    # Each reference H-bond must be detected in at least one direction
    for ref in ref_contacts:
        lig_is_donor, lig_res, lig_name, rec_res, rec_name = ref
        same_dir = ref in test_contacts
        # Also accept the same atom pair detected with swapped donor/acceptor,
        # since the heavy-atom method cannot reliably assign these roles
        swap_dir = (not lig_is_donor, lig_res, lig_name, rec_res, rec_name) in (
            test_contacts
        )
        assert same_dir or swap_dir, f"Missing H-bond: {ref}"
    # Although the hydrogen bonds measured only based on heavy atoms may contain false
    # positives, there should not be excessively many false positives
    unique_pairs = {(c[1], c[2], c[3], c[4]) for c in test_contacts}
    unique_ref_pairs = {(c[1], c[2], c[3], c[4]) for c in ref_contacts}
    assert len(unique_pairs) <= len(unique_ref_pairs) * FALSE_POSITIVE_FACTOR


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
    contact_measurement = peppr.ContactMeasurement(
        receptor, ligand, cutoff=4.0, ph=6.0, use_resonance=use_resonance
    )
    test_contacts = contact_measurement.find_salt_bridges()

    if use_resonance:
        # the expected use should recover all expected contacts
        assert len(test_contacts) == len(expected_contacts)
    else:
        # If resonance is not considered, some contacts may be missed
        assert len(test_contacts) <= len(expected_contacts)

    for i, j in test_contacts:
        contact = (
            receptor.res_id[i].item(),
            receptor.atom_name[i].item(),
            ligand.atom_name[j].item(),
        )
        assert contact in possible_ref_contacts


@pytest.mark.parametrize("use_resonance", [True, False])
@pytest.mark.parametrize("use_tautomers", [True, False])
@pytest.mark.parametrize(
    ["pdb_id", "valid_residues", "valid_ligands", "expected_kinds"],
    [
        ("1acj", {"TRP", "PHE"}, {"THA"}, {struc.PiStacking.PARALLEL}),
        ("7gc7", {"HIS"}, {"L93"}, {struc.PiStacking.PERPENDICULAR}),
    ],
)
def test_find_stacking_interactions(
    pdb_id,
    valid_residues,
    valid_ligands,
    expected_kinds,
    use_resonance,
    use_tautomers,
):
    """Test π-stacking interaction detection between protein and ligand."""
    pdbx_file = pdbx.CIFFile.read(
        Path(__file__).parent / "data" / "pdb" / f"{pdb_id}.cif"
    )
    atoms = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    receptor = atoms[~atoms.hetero]
    ligand = atoms[atoms.hetero]

    contact_measurement = peppr.ContactMeasurement(
        receptor,
        ligand,
        # skip resonance and tautomer enumeration when it is not needed
        use_resonance=use_resonance,
        use_tautomers=use_tautomers,
    )
    interactions = contact_measurement.find_stacking_interactions()
    assert len(interactions) > 0

    for protein_indices, ligand_indices, kind in interactions:
        assert all(
            atom.res_name in valid_residues for atom in receptor[protein_indices]
        ), "Found interaction with unexpected residue"
        assert all(atom.res_name in valid_ligands for atom in ligand[ligand_indices]), (
            "Found interaction with unexpected ligand"
        )
        assert kind in expected_kinds, (
            f"Expected stacking kind in {expected_kinds}, got {kind}"
        )


@pytest.mark.parametrize("use_resonance", [True, False])
def test_find_pi_cation_interactions(use_resonance):
    """Test cation-π interaction detection between protein and ligand.

    Uses PDB 2ack (acetylcholinesterase with edrophonium), which has known
    cation-π interactions between the TRP 84 indole rings and the positively
    charged EDR ligand.
    """
    pdbx_file = pdbx.CIFFile.read(Path(__file__).parent / "data" / "pdb" / "2ack.cif")
    atoms = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    atoms = peppr.standardize(atoms)
    atoms = atoms[atoms.chain_id == struc.get_chains(atoms)[0]]
    receptor = atoms[struc.filter_amino_acids(atoms) & ~atoms.hetero]
    ligand = atoms[(atoms.res_name == "EDR") & atoms.hetero]

    contact_measurement = peppr.ContactMeasurement(
        receptor, ligand, use_resonance=use_resonance
    )
    interactions = contact_measurement.find_pi_cation_interactions()
    assert len(interactions) == 2

    # Both interactions are the 5- and 6-membered TRP 84 rings with the EDR cation
    for receptor_indices, ligand_indices, cation_in_receptor in interactions:
        assert set(receptor[receptor_indices].res_name) == {"TRP"}
        assert set(receptor[receptor_indices].res_id) == {84}
        assert set(ligand[ligand_indices].res_name) == {"EDR"}
        # The cation is in the ligand (EDR), not the receptor
        assert not cation_in_receptor


def test_find_pi_cation_interactions_resonance():
    """Test cation-π interaction detection in SLC19A3 with metformin.

    Uses PDB 8Z7W (SLC19A3-Metformin outward structure). Metformin (MF8)
    has a biguanide group with formal +1 charges on N08 and N05, but
    neither is geometrically positioned above an aromatic ring. With
    resonance, the positive charge delocalizes across all five nitrogens
    in the conjugated system, including N02 which sits above the TYR 113
    ring — enabling detection of one cation-π interaction.
    """
    pdbx_file = pdbx.CIFFile.read(Path(__file__).parent / "data" / "pdb" / "8z7w.cif")
    atoms = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    atoms = peppr.standardize(atoms)
    atoms = atoms[atoms.chain_id == struc.get_chains(atoms)[0]]
    receptor = atoms[struc.filter_amino_acids(atoms) & ~atoms.hetero]
    ligand = atoms[(atoms.res_name == "MF8") & atoms.hetero]

    # Without resonance: N08/N05 have formal charges but aren't above any ring
    cm_no_res = peppr.ContactMeasurement(receptor, ligand, use_resonance=False)
    assert len(cm_no_res.find_pi_cation_interactions()) == 0

    # With resonance: cation-π interaction found
    cm_res = peppr.ContactMeasurement(receptor, ligand, use_resonance=True)
    interactions = cm_res.find_pi_cation_interactions()
    assert len(interactions) == 1

    receptor_indices, ligand_indices, cation_in_receptor = interactions[0]
    assert set(receptor[receptor_indices].res_name) == {"TYR"}
    assert set(receptor[receptor_indices].res_id) == {113}
    assert set(ligand[ligand_indices].res_name) == {"MF8"}
    assert not cation_in_receptor


def test_find_salt_bridges_resonance():
    """Test that resonance enables salt bridge detection in metformin (MF8).

    Uses PDB 8Z7W (SLC19A3-Metformin). At a tight threshold of 3.0 Å,
    without resonance no salt bridge is found: the formally charged OE2 of
    GLU 110 is 3.48 Å from N08. With resonance the other carboxyl oxygen
    OE1 (2.65 Å from N08) is also considered negatively charged, bringing
    the bridge within threshold.
    """
    pdbx_file = pdbx.CIFFile.read(Path(__file__).parent / "data" / "pdb" / "8z7w.cif")
    atoms = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    atoms = peppr.standardize(atoms)
    atoms = atoms[atoms.chain_id == struc.get_chains(atoms)[0]]
    receptor = atoms[struc.filter_amino_acids(atoms) & ~atoms.hetero]
    ligand = atoms[(atoms.res_name == "MF8") & atoms.hetero]

    # Without resonance: no salt bridge at tight threshold
    cm_no_res = peppr.ContactMeasurement(receptor, ligand, use_resonance=False)
    assert len(cm_no_res.find_salt_bridges(threshold=3.0)) == 0

    # With resonance: OE1 of GLU 110 is now also negative -> bridge found
    cm_res = peppr.ContactMeasurement(receptor, ligand, use_resonance=True)
    bridges = cm_res.find_salt_bridges(threshold=3.0)
    assert len(bridges) == 1
    receptor_i, ligand_i = bridges[0]
    assert receptor.res_name[receptor_i] == "GLU"
    assert int(receptor.res_id[receptor_i]) == 110
    assert ligand.res_name[ligand_i] == "MF8"


@pytest.mark.parametrize("use_tauts", [True, False])
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
def test_no_contacts(contact_method, use_tauts):
    """
    Check if the output array shape is still correct if no contacts are found.
    To do this, place the receptor and ligand far apart from each other.
    """
    ligand = info.residue("ALA")
    ligand = ligand[ligand.element != "H"]
    receptor = ligand.copy()
    # Move the receptor far away from the ligand
    receptor.coord += 1000

    contact_measurement = peppr.ContactMeasurement(
        receptor, ligand, use_tautomers=use_tauts
    )
    contacts = contact_method(contact_measurement)
    assert contacts.shape == (0, 2)


@pytest.mark.parametrize("use_tauts", [True, False])
def test_no_contacts_hbonds(use_tauts):
    """Check that find_hbonds returns empty arrays when no contacts exist."""
    ligand = info.residue("ALA")
    ligand = ligand[ligand.element != "H"]
    receptor = ligand.copy()
    receptor.coord += 1000

    contact_measurement = peppr.ContactMeasurement(
        receptor, ligand, use_tautomers=use_tauts
    )
    rec_donates, lig_donates = contact_measurement.find_hbonds()
    assert rec_donates.shape == (0, 2)
    assert lig_donates.shape == (0, 2)


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

    contact_measurement = peppr.ContactMeasurement(receptor, ligand)
    contact_method(contact_measurement)


def test_no_bonds_hbonds():
    """Check that find_hbonds works when atoms have no bonds."""
    ligand = info.residue("ALA")
    ligand = ligand[ligand.element != "H"]
    ligand.bonds = struc.BondList(ligand.array_length())
    receptor = ligand.copy()
    receptor.coord[:, 0] += 5

    contact_measurement = peppr.ContactMeasurement(receptor, ligand)
    contact_measurement.find_hbonds()


def test_find_resonance_charges():
    """
    Test finding charged atoms in resonance structures using a real ligand from a PDB file.
    """
    ligand = info.residue("ASP")
    # standardize removes hydrogens - needed to estimate formal charges
    ligand = peppr.standardize(ligand)

    # set annotations for the benefit of finding charged atoms
    ligand.set_annotation("charge", peppr.estimate_formal_charges(ligand, 7.4))
    # create rdkit ligand object
    ligand_mol = rdkit_interface.to_mol(ligand)
    peppr.sanitize(ligand_mol)

    charged_atoms = np.where(ligand.charge != 0)[0]
    assert np.equal(charged_atoms, [0, 7, 8]).all()

    # get charged atoms and their resonance groups
    pos_mask, neg_mask, ligand_conjugated_groups = peppr.find_resonance_charges(
        ligand_mol
    )
    assert len(set(ligand_conjugated_groups)) < len(ligand_conjugated_groups), (
        "Number of groups should be less than number of atoms as some are conjugated"
    )
    charged_atom_mask = pos_mask | neg_mask
    ligand_charged_in_resonance_atoms = np.where(charged_atom_mask)[0]
    assert set(charged_atoms) != set(ligand_charged_in_resonance_atoms), (
        "Charged atoms do not match those found in resonance structures"
    )
    assert np.equal(ligand_charged_in_resonance_atoms, [0, 3, 6, 7, 8]).all()


@pytest.mark.parametrize(
    ["smiles", "expected_donors", "expected_acceptors"],
    [
        # specified imidazole nHs
        ("c1c[nH]cn1", [2, 4], [2, 4]),
        ("c1cnc[nH]1", [2, 4], [2, 4]),
        # unspecified imidazole symmetric - both nitrogens can be donors and acceptors
        ("c1c[nH]c[nH+]1", [2, 4], []),
        ("c1c[nH+]c[nH]1", [2, 4], []),
        ("c1c[nH]c[nH]1", [2, 4], []),
        # unspecified imidazole asymmetric
        ("Cc1cncn1", [3, 5], [3, 5]),
        # unspecified imidazole symmetric - both nitrogens can be donors and acceptors
        ("c1cncn1", [2, 4], [2, 4]),
    ],
    ids=[
        "specified imidazole nHs 1",
        "specified imidazole nHs 2",
        "specified charged imidazole symmetric 1",
        "specified charged imidazole symmetric 2",
        "specified uncharged imidazole symmetric 1",
        "unspecified imidazole asymmetric 1",
        "unspecified imidazole symmetric 4",
    ],
)
def test_find_tautomeric_hbond_patterns(smiles, expected_donors, expected_acceptors):
    """
    Test if you can find all hydrogen bond donor/acceptor candidates between
    tautomeric forms of the same ligand.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    peppr.sanitize(mol)
    ligand_tautomers = peppr.get_interchangeable_tautomers(mol)

    matched_ligand_donors: set[np.int_] = set()
    for ligand_tautomer in ligand_tautomers:
        matched_ligand_donors |= set(
            peppr.find_atoms_by_pattern(ligand_tautomer, DONOR_PATTERN)
        )
    matched_ligand_donors = np.array(list(matched_ligand_donors))
    assert (matched_ligand_donors == expected_donors).all()

    matched_ligand_acceptors = set()
    for ligand_tautomer in ligand_tautomers:
        matched_ligand_acceptors |= set(
            peppr.find_atoms_by_pattern(ligand_tautomer, ACCEPTOR_PATTERN)
        )
    matched_ligand_acceptors = np.array(list(matched_ligand_acceptors))
    assert (matched_ligand_acceptors == expected_acceptors).all()


@pytest.mark.parametrize("use_tauts", [True, False])
@pytest.mark.parametrize(
    ["pdbid", "chain", "lig", "hbs", "hbs_plus"],
    [
        [
            "7gc7",
            "A",
            "L93",
            [
                # detected easily without tautomers
                (True, 404, "N1", 140, "O"),
                (True, 404, "N1", 166, "OE2"),
                (False, 404, "O1", 166, "N"),
            ],
            # only detected if receptor His163 tautomers are considered
            [(False, 404, "O2", 163, "NE2")],
        ],
    ],
)
def test_tautomer_hbond_detection(use_tauts, pdbid, chain, lig, hbs, hbs_plus):
    """Test that tautomer enumeration recovers additional H-bonds."""
    pdbx_file = pdbx.CIFFile.read(
        Path(__file__).parent / "data" / "pdb" / f"{pdbid}.cif"
    )
    atoms = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    atoms = atoms[atoms.chain_id == chain]
    atoms = peppr.standardize(atoms)

    receptor_mask = ~atoms.hetero
    ligand_mask = atoms.hetero & (atoms.res_name == lig)

    cm = peppr.ContactMeasurement(
        receptor=atoms[receptor_mask],
        ligand=atoms[ligand_mask],
        use_tautomers=use_tauts,
    )

    rec_donates, lig_donates = cm.find_hbonds()
    test_contacts = []
    for lig_is_donor, contacts in [(False, rec_donates), (True, lig_donates)]:
        for receptor_i, ligand_i in contacts:
            test_contacts.append(
                (
                    lig_is_donor,
                    atoms.res_id[ligand_mask][ligand_i],
                    atoms.atom_name[ligand_mask][ligand_i],
                    atoms.res_id[receptor_mask][receptor_i],
                    atoms.atom_name[receptor_mask][receptor_i],
                )
            )

    if use_tauts:
        hbs += hbs_plus
    assert np.array(hb in test_contacts for hb in hbs).all()


def test_sp2_acceptor_widening():
    """Test that SP2 single-neighbor acceptor widening recovers a wide-angle
    H-bond in 3ECA (ASP401:O as acceptor, C-O...N = 158 degrees).

    :meth:`find_hbonds` enables SP2 acceptor widening automatically.
    Without it (i.e. ``find_contacts_by_pattern`` with default parameters),
    the standard [ideal - tolerance, ideal + tolerance] = [90, 150] degree
    range rejects this contact.
    """
    pdbx_file = pdbx.CIFFile.read(Path(__file__).parent / "data" / "pdb" / "3eca.cif")
    atoms = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    atoms = atoms[atoms.chain_id == "A"]
    atoms = peppr.standardize(atoms)

    receptor_mask = ~atoms.hetero
    ligand_mask = atoms.hetero & (atoms.res_name == "ASP")

    cm = peppr.ContactMeasurement(
        receptor=atoms[receptor_mask],
        ligand=atoms[ligand_mask],
    )

    # With widening (via find_hbonds): the wide-angle contact is detected
    rec_donates, lig_donates = cm.find_hbonds()
    test_contacts = []
    for lig_is_donor, contacts in [(False, rec_donates), (True, lig_donates)]:
        for receptor_i, ligand_i in contacts:
            test_contacts.append(
                (
                    lig_is_donor,
                    atoms.res_id[ligand_mask][ligand_i],
                    atoms.atom_name[ligand_mask][ligand_i],
                    atoms.res_id[receptor_mask][receptor_i],
                    atoms.atom_name[receptor_mask][receptor_i],
                )
            )

    expected = [
        (True, 401, "N", 59, "OE1"),
        (True, 401, "N", 90, "OD2"),
        (False, 401, "OD2", 12, "N"),
        (False, 401, "OXT", 58, "N"),
        (False, 401, "OD2", 89, "N"),
        # Wide-angle acceptor (C-O...N = 158 deg)
        (False, 401, "O", 90, "N"),
    ]
    assert np.array(hb in test_contacts for hb in expected).all()

    # Without widening: the wide-angle contact is rejected
    narrow_contacts = []
    for rp, lp in [
        (ACCEPTOR_PATTERN, DONOR_PATTERN),
        (DONOR_PATTERN, ACCEPTOR_PATTERN),
    ]:
        for ri, li in cm.find_contacts_by_pattern(
            rp,
            lp,
            HBOND_DISTANCE_SCALING,
        ):
            narrow_contacts.append(
                (
                    lp == DONOR_PATTERN,
                    atoms.res_id[ligand_mask][li],
                    atoms.atom_name[ligand_mask][li],
                    atoms.res_id[receptor_mask][ri],
                    atoms.atom_name[receptor_mask][ri],
                )
            )
    wide_angle_hb = (False, 401, "O", 90, "N")
    assert wide_angle_hb not in narrow_contacts


def test_full_vs_residue_tautomer_enumeration():
    """
    Test that residue-wise tautomer enumeration provides consistent results with full enumeration
    """
    pdbx_file = pdbx.CIFFile.read(Path(__file__).parent / "data" / "pdb" / "7gc7.cif")
    atoms = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    atoms = atoms[atoms.chain_id == "A"]
    atoms = peppr.standardize(atoms)

    receptor_mask = ~atoms.hetero
    ligand_mask = (atoms.hetero) & (atoms.res_name != "DMS")

    contact_measure = peppr.ContactMeasurement(
        receptor=atoms[receptor_mask],
        ligand=atoms[ligand_mask],
        cutoff=8.0,
        use_tautomers=False,
    )

    for receptor_pattern in [DONOR_PATTERN, ACCEPTOR_PATTERN]:
        # using explicit enumeration
        full_tautomers = peppr.get_interchangeable_tautomers(
            contact_measure._binding_site_mol
        )
        matched_indices_full = set()
        for tautomer in full_tautomers:
            matched_indices_full |= set(
                peppr.find_atoms_by_pattern(tautomer, receptor_pattern)
            )
        matched_indices_full = np.array(sorted(matched_indices_full), dtype=int)
        # Per-residue-instance enumeration with SMILES-based dedup
        residue_map = peppr.contacts.ResidueMap(contact_measure._binding_site_mol)
        matched_indices_frags = residue_map.find_tautomer_atoms_by_pattern(
            receptor_pattern
        )
        assert (matched_indices_full == matched_indices_frags).all()
