import itertools
import string
from math import factorial
from pathlib import Path
import biotite.interface.rdkit as rdkit_interface
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest
import rdkit.Chem.AllChem as Chem
import peppr
from tests.common import list_test_pdb_files


@pytest.fixture
def bromochlorofluoromethane():
    """
    Get the structure of Bromochlorofluoromethane (BCF), a chiral molecule with
    minimum size.
    """
    # Make sure to use SMILES that determines the enantiomer
    mol = Chem.MolFromSmiles("[C@H](F)(Cl)Br")
    # RDKit uses implicit hydrogen atoms by default, but Biotite requires explicit ones
    mol = Chem.AddHs(mol)
    # Create a 3D conformer
    conformer_id = Chem.EmbedMolecule(mol)
    Chem.UFFOptimizeMolecule(mol)
    return rdkit_interface.from_mol(mol, conformer_id)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("crop", [False, True])
@pytest.mark.parametrize("pdb_id", list_test_pdb_files(), ids=lambda path: path.stem)
@pytest.mark.parametrize("use_heuristic", [False, True])
def test_matching_atoms(pdb_id, crop, seed, use_heuristic):
    """
    Check for each of the input oligomers, whether the original atom order is recovered
    via :func:`find_optimal_match()`, if the chain order is shuffled.
    """
    pdbx_file = pdbx.CIFFile.read(pdb_id)
    reference = pdbx.get_structure(
        pdbx_file,
        model=1,
        include_bonds=True,
        # Use label chain IDs to ensure each small molecule gets a separate chain ID
        use_author_fields=False,
    )
    reference = peppr.standardize(reference)
    # Annotate small molecules as hetero
    reference.hetero[~struc.filter_amino_acids(reference)] = True
    _annotate_atom_order(reference)

    chains = list(struc.chain_iter(reference))
    if not use_heuristic:
        # To avoid combinatorial explosion, only use every second chain
        chains = [chain for i, chain in enumerate(chains) if i % 2 == 0]
        reference = struc.concatenate(chains)
    if crop:
        # Remove the the first and last residue of protein chains,
        # to check if also similar chains are matched correctly
        chains = [
            chain[~np.isin(chain.res_id, (chain.res_id[0], chain.res_id[-1]))]
            if struc.get_residue_count(chain) > 1
            else chain
            for chain in chains
        ]
    # Randomly swap chains
    rng = np.random.default_rng(seed)
    chains = [chains[i] for i in rng.permutation(len(chains))]
    pose = struc.concatenate(chains)

    matched_reference, matched_pose = peppr.find_optimal_match(
        reference, pose, use_heuristic=use_heuristic
    )

    _check_match(matched_reference, matched_pose, reference, pose)
    # The pose is simply the reference with permuted chains
    # Hence, after reordering the distances should be 0
    matched_reference, matched_pose = peppr.filter_matched(
        matched_reference, matched_pose
    )
    matched_pose, _ = struc.superimpose(matched_reference, matched_pose)
    assert np.allclose(struc.distance(matched_reference, matched_pose), 0.0, atol=1e-4)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("use_different_entities", [False, True])
@pytest.mark.parametrize("n_chains", (1, 2, 4))
@pytest.mark.parametrize("use_heuristic", [False, True])
def test_matching_chains(n_chains, seed, use_different_entities, use_heuristic):
    """
    Check for a fake homomeric complex of two small molecules that
    :func:`find_optimal_match()` finds corresponding chains.
    For this purpose create a system containing randomly translated copies of the same
    chain.
    For the pose translations are slightly perturbed.
    Then the chains in the pose are randomly swapped and it is expected that
    :func:`find_optimal_match()` is able to restore the original order.

    Also check if permutation of two chains is forbidden if they are a different entity,
    by assigning each chain a different entity and thus forbid permuting any chain
    """
    SYSTEM_DISTANCE = 1000
    CHAIN_DISTANCE = 10
    PERTURBATION_DISTANCE = 0.1

    rng = np.random.default_rng(seed)
    monomer = struc.info.residue("ALA")
    # Remove hydrogen atoms to make the test example smaller
    monomer = peppr.standardize(monomer)
    # Create the homomeric 'complex'
    reference_chains = [
        struc.translate(monomer, rng.normal(scale=CHAIN_DISTANCE, size=3))
        for _ in range(n_chains)
    ]
    for chain, chain_id in zip(reference_chains, string.ascii_uppercase):
        # Assign a unique ID to each chain
        chain.chain_id[:] = chain_id
    if use_different_entities:
        # Enforce that each chain becomes a different entity
        for i, chain in enumerate(reference_chains):
            chain.res_name[:] = f"LIG{i}"
            chain.hetero[:] = True

    # Slightly perturb the reference coordinates to obtain the pose
    pose_chains = [
        struc.translate(chain, rng.normal(scale=PERTURBATION_DISTANCE, size=3))
        for chain in reference_chains
    ]
    # As the perturbation is much smaller than the chain distance,
    # this RMSD should be the smallest
    ref_rmsd = struc.rmsd(
        struc.concatenate(reference_chains), struc.concatenate(pose_chains)
    )

    swap_indices = rng.permutation(n_chains)
    reference = struc.concatenate(reference_chains)
    # Move the entire pose
    # to check if the tested function uses superimposition properly
    pose = struc.translate(
        struc.concatenate([pose_chains[i] for i in swap_indices]),
        rng.normal(scale=SYSTEM_DISTANCE, size=3),
    )
    if use_different_entities:
        # Keep the original entity order
        pose.res_name = reference.res_name.copy()

    matched_reference, matched_pose = peppr.find_optimal_match(
        reference, pose, use_heuristic=use_heuristic
    )

    _check_match(matched_reference, matched_pose, reference, pose)
    if use_different_entities:
        # Permutation across molecules is totally forbidden, as they belong to different
        # entities
        assert matched_pose.res_name.tolist() == matched_reference.res_name.tolist()
    else:
        # Permutation is allowed
        # -> expect that the original minimum RMSD is achieved]
        superimposed_pose, _ = struc.superimpose(matched_reference, matched_pose)
        assert struc.rmsd(matched_reference, superimposed_pose) <= ref_rmsd


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("use_heuristic", [False, True])
@pytest.mark.parametrize(
    ["ref_monomer_multiplicity", "pose_monomer_multiplicity", "should_fail"],
    [
        # The fake complex has two different chains, one with 2 monomers and one with 4
        ((2, 4), (2, 4), False),  # Trivial case all chains can be matched
        ((1, 4), (2, 4), False),  # One chain is missing in one structure
        ((1, 4), (2, 3), False),  # One chain is missing in either structure
        ((0, 4), (2, 4), False),  # One entity is completely absent in one structure
        ((0, 4), (2, 0), True),  # No chain can be matched
    ],
)
def test_partial_matching_chains(
    ref_monomer_multiplicity,
    pose_monomer_multiplicity,
    should_fail,
    use_heuristic,
    swap,
    shuffle,
):
    """
    Check corresponding chains can be matched, if either structure misses some chains.
    To do this create a fake complex and selectively remove chains in the reference
    and pose.
    Expect that still corresponding chains are found.
    """
    MAX_CHAIN_DISTANCE = 100
    COMP_NAMES = ["GLY", "ALA"]

    if swap:
        ref_monomer_multiplicity, pose_monomer_multiplicity = (
            pose_monomer_multiplicity,
            ref_monomer_multiplicity,
        )
    max_multiplicity = tuple(
        [
            max(ref_mult, pose_mult)
            for ref_mult, pose_mult in zip(
                ref_monomer_multiplicity, pose_monomer_multiplicity, strict=True
            )
        ]
    )

    rng = np.random.default_rng(0)
    reference_chains = []
    pose_chains = []
    atom_id_offset = 0
    chain_id_generator = iter(string.ascii_uppercase)
    for i, (multiplicity, comp_name) in enumerate(zip(max_multiplicity, COMP_NAMES)):
        monomer = info.residue(comp_name)
        monomer = peppr.standardize(monomer)
        _annotate_atom_order(monomer)
        for j in range(multiplicity):
            chain = monomer.copy()
            chain.atom_id += atom_id_offset
            chain.chain_id[:] = next(chain_id_generator)
            chain.coord += rng.uniform(-MAX_CHAIN_DISTANCE, MAX_CHAIN_DISTANCE, size=3)
            # Only keep the chain in the reference/pose,
            # if the desired multiplicity is not reached yet
            if j < ref_monomer_multiplicity[i]:
                reference_chains.append(chain)
            if j < pose_monomer_multiplicity[i]:
                pose_chains.append(chain)
            atom_id_offset += monomer.array_length()
    if shuffle:
        order = rng.permutation(len(pose_chains))
        pose_chains = [pose_chains[i] for i in order]
    reference = struc.concatenate(reference_chains)
    pose = struc.concatenate(pose_chains)

    if should_fail:
        with pytest.raises(peppr.UnmappableEntityError):
            matched_reference, matched_pose = peppr.find_optimal_match(
                reference,
                pose,
                use_heuristic=use_heuristic,
                allow_unmatched_entities=True,
            )
    else:
        matched_reference, matched_pose = peppr.find_optimal_match(
            reference, pose, use_heuristic=use_heuristic, allow_unmatched_entities=True
        )
        _check_match(matched_reference, matched_pose, reference, pose)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("use_heuristic", [False, True])
@pytest.mark.parametrize("omit_some_atoms", [False, True])
@pytest.mark.parametrize("pdb_id", list_test_pdb_files(), ids=lambda path: path.stem)
def test_shuffled_atom_order(pdb_id, omit_some_atoms, use_heuristic, seed):
    """
    Check if the corresponding atom order can be regained, even if the atoms within the
    same residue are not in the same order and some atoms are potentially missing.
    """
    P_REMOVED = 0.05

    pdbx_file = pdbx.CIFFile.read(pdb_id)
    reference = pdbx.get_structure(
        pdbx_file,
        model=1,
        include_bonds=True,
        # Use label chain IDs to ensure each small molecule gets a separate chain ID
        use_author_fields=False,
    )
    # Remove salt and water
    reference = reference[
        ~struc.filter_solvent(reference) & ~struc.filter_monoatomic_ions(reference)
    ]
    # Limit test to a single chain
    reference = next(struc.chain_iter(reference))
    _annotate_atom_order(reference)
    pose = reference.copy()

    rng = np.random.default_rng(seed)
    if omit_some_atoms:
        reference = reference[
            rng.choice(
                (False, True),
                p=(P_REMOVED, 1 - P_REMOVED),
                size=reference.array_length(),
            )
        ]
        pose = pose[
            rng.choice(
                (False, True), p=(P_REMOVED, 1 - P_REMOVED), size=pose.array_length()
            )
        ]
    # Make sure only the order within each residue is shuffled, not the global order
    orders_within_residue = []
    for start, stop in itertools.pairwise(
        struc.get_residue_starts(pose, add_exclusive_stop=True)
    ):
        orders_within_residue.append(rng.permutation(range(start, stop)))
    shuffled_order = np.concatenate(orders_within_residue)
    pose = pose[shuffled_order]

    matched_reference, matched_pose = peppr.find_optimal_match(
        reference, pose, use_heuristic=use_heuristic
    )
    # Ensure that all applicable atoms were matched
    if omit_some_atoms:
        min_matching_atoms = (1 - 2 * P_REMOVED) * reference.array_length()
    else:
        min_matching_atoms = reference.array_length()
    _check_match(matched_reference, matched_pose, reference, pose, min_matching_atoms)
    # Ensure that the atoms were reordered only within each residue and not globally
    # -> the residue IDs should be monotonically increasing
    assert np.all(np.diff(matched_pose.res_id) >= 0)


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize(
    ["comp_name", "swap_atom_names"],
    [
        # Ethanol: No symmetries -> no coordinates to swap
        ("EOH", []),
        # Alanine: Both carboxy oxygen atoms are equivalent
        #          despite having different formal bond orders
        ("ALA", [("O", "OXT")]),
        # Phenol: Symmetry along the hydroxyl axis
        #         The additional challenge is that the 'equivalent' bond types are
        #         different in kekulized form, as a single aromatic bond faces a double
        #         aromatic bond
        ("IPH", [("C2", "C6"), ("C3", "C5")]),
        # Trichloromethane: Chlorine atoms are equivalent
        ("MCH", [("CL1", "CL2")]),
    ],
)
@pytest.mark.parametrize("use_heuristic", [False, True])
def test_matching_small_molecules(comp_name, swap_atom_names, shuffle, use_heuristic):
    """
    Check for known molecules with symmetries whether equivalent atoms are detected
    in :func:`_find_optimal_molecule_permutation()`.
    For this purpose swap coordinates of equivalent atoms and expect that
    :func:`find_optimal_match` swaps them back in order to minimize the RMSD.

    Furthermore shuffle the atom order of the `AtomArray` to increase the difficulty.
    """
    TRANSLATION_VEC = np.array([10, 20, 30])

    reference = struc.info.residue(comp_name)
    # Hydrogen atoms are not considered
    reference = reference[reference.element != "H"]
    # Mark as small molecule
    reference.hetero[:] = True
    _annotate_atom_order(reference)
    pose = reference.copy()
    for atom_name1, atom_name2 in swap_atom_names:
        indices = np.where(np.isin(pose.atom_name, (atom_name1, atom_name2)))[0]
        pose.coord[indices] = pose.coord[indices[::-1]]
        pose.atom_id[indices] = pose.atom_id[indices[::-1]]
    if shuffle:
        # Completely change the order of atoms in the AtomArray
        rng = np.random.default_rng(seed=0)
        pose = pose[rng.permutation(pose.array_length())]

    # To increase difficulty, the atoms are translated -> superimposition is necessary
    pose.coord += TRANSLATION_VEC

    matched_reference, matched_pose = peppr.find_optimal_match(
        reference, pose, use_heuristic=use_heuristic
    )

    _check_match(matched_reference, matched_pose, reference, pose)
    # After superimposition of corresponding atoms,
    # the pose should perfectly overlap with the reference again
    matched_pose, _ = struc.superimpose(matched_reference, matched_pose)
    assert struc.rmsd(matched_reference, matched_pose) == pytest.approx(0.0, abs=1e-4)


@pytest.mark.filterwarnings("error::peppr.GraphMatchWarning")
@pytest.mark.parametrize("use_heuristic", [False, True])
def test_match_kekulized_to_aromatic(use_heuristic):
    """
    Check if an aromatic molecule can be matched to the same molecule where its
    bonds are kekulized.
    """
    # A molecule without symmetry
    COMP_NAME = "HIS"

    reference = struc.info.residue(COMP_NAME)
    reference = reference[reference.element != "H"]
    reference.hetero[:] = True
    _annotate_atom_order(reference)

    pose = reference.copy()
    # Kekulize the bonds
    bond_array = pose.bonds.as_array()
    for aromatic_bond_type, kelulized_bond_type in [
        (struc.BondType.AROMATIC_SINGLE, struc.BondType.SINGLE),
        (struc.BondType.AROMATIC_DOUBLE, struc.BondType.DOUBLE),
        (struc.BondType.AROMATIC_TRIPLE, struc.BondType.TRIPLE),
    ]:
        mask = bond_array[:, 2] == aromatic_bond_type
        bond_array[mask, 2] = kelulized_bond_type
    pose.bonds = struc.BondList(pose.array_length(), bond_array)

    # Shuffle atom order to increase difficulty
    rng = np.random.default_rng(seed=0)
    pose = pose[rng.permutation(pose.array_length())]

    matched_reference, matched_pose = peppr.find_optimal_match(
        reference, pose, use_heuristic=use_heuristic
    )

    _check_match(matched_reference, matched_pose, reference, pose)


@pytest.mark.xfail(
    reason="There is a fallback with 'useChirality=False', whichmakes this test fail"
)
@pytest.mark.parametrize("use_heuristic", [False, True])
@pytest.mark.parametrize("mirror", [False, True])
def test_no_matching_of_enantiomers(bromochlorofluoromethane, mirror, use_heuristic):
    """
    Ensure that :func:`_find_optimal_molecule_permutation()` does not map enantiomers
    to each other.
    To achieve this, simply create a fake molecule that contains a compound and its
    mirror image.

    As positive control do the same for two copies of a compound within the same
    molecule and expect them to be mapped to each other.
    """
    TRANSLATION_VEC = np.array([10, 20, 30])

    mol_1 = bromochlorofluoromethane
    mol_1.hetero[:] = True
    mol_1.chain_id[:] = "A"
    mol_2 = mol_1.copy()
    mol_2.chain_id[:] = "B"
    if mirror:
        # Mirror at the x-axis
        mol_2.coord[:, 0] *= -1
    # Move them a bit apart
    reference = mol_1 + struc.translate(mol_2, TRANSLATION_VEC)
    _annotate_atom_order(reference)
    # For the pose move the other coordinates,
    # so that :func:`_find_optimal_molecule_permutation()` would normally simply
    # exchange the coordinates of the two molecules to minimize the RMSD
    pose = struc.translate(mol_1, TRANSLATION_VEC) + mol_2
    _annotate_atom_order(pose)

    matched_reference, matched_pose = peppr.find_optimal_match(
        reference, pose, use_heuristic=use_heuristic
    )

    if mirror:
        # Mapping enantiomers onto each other is not allowed
        # -> the matched atom order should be the same as in the input
        _check_match(matched_reference, matched_pose, reference, pose)
    else:
        # Positive control: Mapping two copies of a compound onto each other is allowed
        # As swapping would also minimize the RMSD, swap back and check if the original
        # atom order is restored
        swapped_pose = struc.concatenate(
            (
                matched_pose[matched_pose.shape[0] // 2 :],
                matched_pose[: matched_pose.shape[0] // 2],
            )
        )
        _check_match(matched_reference, swapped_pose, reference, pose)


@pytest.mark.parametrize("use_heuristic", [True, False])
def test_unmatchable_molecules(use_heuristic):
    """
    Even if two small molecules cannot be matched due to different bond graphs,
    there is a fallback that simply matches atoms in the order of their appearance.
    Test this by deliberately creating two molecules with incompatible bonds
    """
    reference = struc.info.residue("PNN")
    reference = reference[reference.element != "H"]
    _annotate_atom_order(reference)
    pose = reference.copy()
    # Remove bonds in one structure to make the bond graphs incompatible
    pose.bonds = struc.BondList(pose.array_length())

    with pytest.warns(peppr.GraphMatchWarning, match="Incompatible bond graph"):
        matched_reference, matched_pose = peppr.find_optimal_match(
            reference, pose, use_heuristic=use_heuristic
        )

    # As fallback the atoms are matched in the order of their appearance
    _check_match(matched_reference, matched_pose, reference, pose)


def test_exhaustive_mappings():
    """
    Check if :func:`find_all_matches()` finds all possible atom mappings
    for a known example.
    """
    # Hemoglobin: 2*2 equivalent protein chains and 4 heme molecules
    N_MAPPINGS = (
        # Mappings between alpha or beta chains
        factorial(2) ** 2
        # Mappings between heme molecules
        * factorial(4)
        # Each heme molecule has a two carboxy groups with two equivalent oxygen atoms
        * factorial(2) ** (4 * 2)
    )

    pdbx_file = pdbx.CIFFile.read(Path(__file__).parent / "data" / "pdb" / "1a3n.cif")
    reference = pdbx.get_structure(
        pdbx_file,
        model=1,
        include_bonds=True,
        # Use label chain IDs to ensure each small molecule gets a separate chain ID
        use_author_fields=False,
    )
    reference = peppr.standardize(reference)
    _annotate_atom_order(reference)
    pose = reference.copy()

    # Only save the matched `atom_id` annotation for each match,
    # as otherwise a lot of memory would be used for the test
    # Also use a memory efficient (and also hashable) representation of the integers
    # -> use 'tobytes()'
    test_mappings = [
        (
            matched_reference.atom_id.tobytes(),
            matched_pose.atom_id.tobytes(),
        )
        for matched_reference, matched_pose in peppr.find_all_matches(reference, pose)
    ]

    assert len(test_mappings) == N_MAPPINGS
    # All mappings should be unique
    assert len(set(test_mappings)) == len(test_mappings)


@pytest.mark.parametrize("use_structure_match", [True, False])
@pytest.mark.parametrize(
    ["same_residue_name", "same_bond_graph"],
    [
        (False, True),
        (True, False),
    ],
)
def test_small_molecule_entities(
    use_structure_match, same_residue_name, same_bond_graph
):
    """
    If ``use_structure_match=True``, the small molecules should be matched by structure
    matching, irrespective of the residue name, and the other way around if
    ``use_structure_match=False``.

    Test this by matching molecules with the same residue name but different bond graph,
    and vice versa.
    """
    reference = struc.info.residue("LEU")
    reference.hetero[:] = True
    reference = reference[reference.element != "H"]
    if same_bond_graph:
        pose = reference.copy()
    else:
        pose = struc.info.residue("ILE")
        pose.hetero[:] = True
        pose = pose[pose.element != "H"]

    if same_residue_name:
        pose.res_name[:] = reference.res_name
    else:
        pose.res_name[:] = "ILE"

    if (use_structure_match and same_bond_graph) or (
        not use_structure_match and same_residue_name
    ):
        matched_reference, matched_pose = peppr.find_optimal_match(
            reference, pose, use_structure_match=use_structure_match
        )
        assert matched_reference.matched.all()
        assert matched_pose.matched.all()
    else:
        with pytest.raises(peppr.UnmappableEntityError):
            matched_reference, matched_pose = peppr.find_optimal_match(
                reference, pose, use_structure_match=use_structure_match
            )


def _annotate_atom_order(atoms):
    """
    Add the ``atom_id`` annotation to the given atoms for later comparison
    with the :func:`_check_match()` function.
    """
    atoms.set_annotation("atom_id", np.arange(atoms.array_length()))


def _check_match(
    matched_reference,
    matched_pose,
    original_reference,
    original_pose,
    min_matching_atoms=None,
):
    """
    Check if the given atom match is correct.
    If both `matched_reference` and `matched_pose` have the ``atom_id`` annotation,
    it is checked, if the ``matched`` part of the `atom_id` is identical.
    """
    # No atom is removed during the matching...
    assert matched_reference.array_length() == original_reference.array_length()
    assert matched_pose.array_length() == original_pose.array_length()
    # ...until they are filtered to the matched atoms
    matched_reference, matched_pose = peppr.filter_matched(
        matched_reference, matched_pose
    )
    if min_matching_atoms is not None:
        assert len(matched_reference) >= min_matching_atoms
        assert len(matched_pose) >= min_matching_atoms
    if (
        "atom_id" in matched_reference.get_annotation_categories()
        and "atom_id" in matched_pose.get_annotation_categories()
    ):
        assert matched_reference.atom_id.tolist() == matched_pose.atom_id.tolist()
        # Each atom may only appear once
        assert len(np.unique(matched_reference.atom_id)) == len(
            matched_reference.atom_id
        )
        assert len(np.unique(matched_pose.atom_id)) == len(matched_pose.atom_id)
