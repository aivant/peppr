import itertools
import string
from math import factorial
from pathlib import Path
import biotite.interface.rdkit as rdkit_interface
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest
import rdkit.Chem.AllChem as Chem
import peppr
from peppr.match import _all_global_mappings as all_global_mappings
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

    # Use heuristic in this test, as without it the number of possible mappings explodes
    reference_order, pose_order = peppr.find_optimal_match(
        reference, pose, use_heuristic=True
    )
    reordered_reference = reference[reference_order]
    reordered_pose = pose[pose_order]
    reordered_pose, _ = struc.superimpose(reordered_reference, reordered_pose)

    # Each atom may only appear once
    assert len(np.unique(reference_order)) == len(reference_order)
    assert len(np.unique(pose_order)) == len(pose_order)
    # Expect all atoms to be contained in the reordered structures
    # (except for the ones that were cropped)
    n_atoms = min(reference.array_length(), pose.array_length())
    assert reordered_pose.array_length() == n_atoms
    assert reordered_reference.array_length() == n_atoms
    # The pose is simply the reference with permuted chains
    # Hence, after reordering the distances should be 0
    assert np.allclose(
        struc.distance(reordered_reference, reordered_pose), 0.0, atol=1e-4
    )


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("use_different_entities", [False, True])
@pytest.mark.parametrize("n_chains", (1, 2, 4, 8))
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

    reference_order, pose_order = peppr.find_optimal_match(
        reference, pose, use_heuristic=use_heuristic
    )

    # Since no atom is missing in either structure,
    # the reference order should be simply all atoms in the original order
    assert reference_order.tolist() == list(range(len(reference)))
    # Each atom may only appear once
    assert len(np.unique(pose_order)) == len(pose_order)
    if use_different_entities:
        # Permutation across molecules is totally forbidden, as they belong to different
        # entities
        assert pose.res_name[pose_order].tolist() == reference.res_name.tolist()
    else:
        # Permutation is allowed
        # -> expect that the original minimum RMSD is achieved
        pose = pose[pose_order]
        superimposed_pose, _ = struc.superimpose(reference, pose)
        assert struc.rmsd(reference, superimposed_pose) <= ref_rmsd


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("omit_some_atoms", [False, True])
@pytest.mark.parametrize("pdb_id", list_test_pdb_files(), ids=lambda path: path.stem)
def test_shuffled_atom_order(pdb_id, omit_some_atoms, seed):
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

    reference_order, pose_order = peppr.find_optimal_match(reference, pose)
    reordered_reference = reference[reference_order]
    reordered_pose = pose[pose_order]

    # Each atom may only appear once
    assert len(np.unique(reference_order)) == len(reference_order)
    assert len(np.unique(pose_order)) == len(pose_order)
    # Ensure that all applicable atoms were matched
    if omit_some_atoms:
        min_matching_atoms = (1 - 2 * P_REMOVED) * reference.array_length()
    else:
        min_matching_atoms = reference.array_length()
    assert len(pose_order) >= min_matching_atoms
    assert len(reference_order) >= min_matching_atoms
    # Ensure that the correct atoms were matched
    assert np.all(reordered_reference.atom_name == reordered_pose.atom_name)
    assert np.all(reordered_reference.res_name == reordered_pose.res_name)
    assert np.all(reordered_reference.res_id == reordered_pose.res_id)
    # Ensure that the atoms were reordered only within each residue and not globally
    # -> the residue IDs should be monotonically increasing
    assert np.all(np.diff(reordered_pose.res_id) >= 0)


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
def test_matching_small_molecules(comp_name, swap_atom_names, shuffle):
    """
    Check for known molecules with symmetries whether equivalent atoms are detected
    in :func:`_find_optimal_molecule_permutation()`.
    For this purpose swap coordinates of equivalent atoms and expect that
    :func:`find_optimal_match` swaps them back in order to minimize the RMSD.

    Furthermore shuffle the atom order of the AtomArray to increase the difficulty.
    """
    TRANSLATION_VEC = np.array([10, 20, 30])

    reference = struc.info.residue(comp_name)
    # Hydrogen atoms are not considered
    reference = reference[reference.element != "H"]
    # Mark as small molecule
    reference.hetero[:] = True
    pose = reference.copy()
    for atom_name1, atom_name2 in swap_atom_names:
        indices = np.where(np.isin(pose.atom_name, (atom_name1, atom_name2)))[0]
        pose.coord[indices] = pose.coord[indices[::-1]]
    if shuffle:
        # Completely change the order of atoms in the AtomArray
        rng = np.random.default_rng(seed=0)
        pose = pose[rng.permutation(pose.array_length())]

    # To increase difficulty, the atoms are translated -> superimposition is necessary
    pose.coord += TRANSLATION_VEC

    reference_order, pose_order = peppr.find_optimal_match(reference, pose)
    # Translate back
    pose.coord -= TRANSLATION_VEC
    # Apply the order to revert the swapping
    reference = reference[reference_order]
    pose = pose[pose_order]

    # Each atom may only appear once
    assert len(np.unique(reference_order)) == len(reference_order)
    assert len(np.unique(pose_order)) == len(pose_order)
    # Only atoms of the same element can be mapped to each other
    assert pose.element.tolist() == reference.element.tolist()
    # After swapping back the pose should perfectly overlap with the reference again
    assert struc.rmsd(reference, pose) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("mirror", [False, True])
def test_no_matching_of_enantiomers(bromochlorofluoromethane, mirror):
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
    mol_2 = mol_1.copy()
    if mirror:
        # Mirror at the x-axis
        mol_2.coord[:, 0] *= -1
    # Move them a bit apart
    reference = mol_1 + struc.translate(mol_2, TRANSLATION_VEC)
    # For the pose move the other coordinates,
    # so that :func:`_find_optimal_molecule_permutation()` would normally simply
    # exchange the coordinates of the two molecules to minimize the RMSD
    pose = struc.translate(mol_1, TRANSLATION_VEC) + mol_2
    reference_order, pose_order = peppr.find_optimal_match(reference, pose)
    if mirror:
        # Mapping enantiomers onto each other is not allowed
        assert pose_order.tolist() == reference_order.tolist()
    else:
        # Positive control: Mapping two copies of a compound onto each other is allowed
        assert (
            np.concatenate(
                [
                    # Swap the indices for the first and second copy
                    pose_order[: pose_order.shape[0] // 2],
                    pose_order[pose_order.shape[0] // 2 :],
                ]
            ).tolist()
            != reference_order.tolist()
        )


def test_unmatchable_molecules():
    """
    Even if two small molecules cannot be matched, due to different bond graphs,
    there is a fallback that simply matches atoms in the order of their appearance.
    Test this by deliberately creating two molecules with incompatible bonds
    """
    reference = struc.info.residue("PNN")
    reference = reference[reference.element != "H"]
    pose = reference.copy()
    # Remove bonds in one structure to make the bond graphs incompatible
    pose.bonds = struc.BondList(pose.array_length())

    with pytest.warns(peppr.GraphMatchWarning, match="RDKit failed atom matching"):
        reference_order, pose_order = peppr.find_optimal_match(reference, pose)

    # As fallback the atoms are matched in the order of their appearance
    assert np.all(reference_order == np.arange(reference.array_length()))
    assert np.all(pose_order == np.arange(pose.array_length()))


def test_exhaustive_mappings():
    """
    Check if ::func:`_all_global_mappings()` finds all possible atom mappings
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
    pose = reference.copy()
    reference_chains = list(struc.chain_iter(reference))
    pose_chains = list(struc.chain_iter(pose))
    test_mappings = list(all_global_mappings(reference_chains, pose_chains))

    assert len(test_mappings) == N_MAPPINGS

    # All mappings should be unique
    test_mappings_set = set(
        [
            (tuple(ref_indices.tolist()), tuple(pose_indices.tolist()))
            for ref_indices, pose_indices in test_mappings
        ]
    )
    assert len(test_mappings_set) == len(test_mappings)
