__all__ = ["GraphMatchWarning", "find_matching_atoms"]


import itertools
import warnings
from typing import Any
import biotite.interface.rdkit as rdkit_interface
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.structure as struc
import numpy as np
from numpy.typing import NDArray
from rdkit.Chem import (
    AssignStereochemistryFrom3D,
    BondType,
    Mol,
    SanitizeFlags,
    SanitizeMol,
)
from peppr.common import is_small_molecule

# To match atoms between pose and reference chain,
# the residue and atom name are sufficient to unambiguously identify an atom
_ANNOTATIONS_FOR_ATOM_MATCHING = ["res_name", "atom_name"]
_IDENTITY_MATRIX = align.SubstitutionMatrix(
    seq.ProteinSequence.alphabet,
    seq.ProteinSequence.alphabet,
    np.eye(len(seq.ProteinSequence.alphabet), dtype=np.int32),
)


class GraphMatchWarning(UserWarning):
    """
    This warning is raised, if the RDKit based molecule matching fails.
    In this case small molecule reordering is skipped.
    """

    pass


def find_matching_atoms(
    reference: struc.AtomArray,
    pose: struc.AtomArray,
    min_sequence_identity: float = 0.95,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Find the optimal atom order for each pose that minimizes the RMSD to the reference.

    Parameters
    ----------
    reference : struc.AtomArray, shape=(n,)
        The reference structure.
    pose : struc.AtomArray, shape=(n,)
        The pose structure.
    min_sequence_identity : float
        The minimum sequence identity between two chains to be considered the same
        entity.

    Returns
    -------
    reference_order, : np.array, shape=(n,), dtype=int
        The atom order that should be applied to `reference`.
    pose_order : np.array, shape=(n,), dtype=int
        The atom order that should be applied to `pose`.

    Notes
    -----
    For finding the optimal chain permutation, the function uses the
    algorithm described in [1]_.

    References
    ----------
    .. [1] *Protein complex prediction with AlphaFold-Multimer*, Section 7.3, https://doi.org/10.1101/2021.10.04.463034
    """
    ref_chain_starts = struc.get_chain_starts(reference, add_exclusive_stop=True)
    mod_chain_starts = struc.get_chain_starts(pose, add_exclusive_stop=True)
    reference_chains = [
        reference[start:stop] for start, stop in itertools.pairwise(ref_chain_starts)
    ]
    pose_chains = [
        pose[start:stop] for start, stop in itertools.pairwise(mod_chain_starts)
    ]

    # Find corresponding chains by identifying the chain permutation minimizing the RMSD
    mod_chain_order, transform, anchor_index = _find_matching_chain_permutation(
        reference_chains, pose_chains, min_sequence_identity
    )

    # Based on the matching chains, find corresponding atoms within each pair of chains
    ref_atom_orders: list[np.ndarray | None] = [None] * len(reference_chains)
    mod_atom_orders: list[np.ndarray | None] = [None] * len(reference_chains)
    for ref_i, mod_i in zip(np.arange(len(reference_chains)), mod_chain_order):
        if is_small_molecule(reference_chains[ref_i]):
            try:
                ref_atom_orders[ref_i], mod_atom_orders[mod_i] = (
                    _find_optimal_molecule_permutation(
                        reference_chains[ref_i],
                        transform.apply(pose_chains[mod_i]),
                        # If no other chain already determined the optimal transform,
                        # superimpose each permuted molecule to get the best RMSD
                        superimpose=True if mod_i == anchor_index else False,
                    )
                )
            except Exception as e:
                warnings.warn(
                    f"RDKit failed atom matching: {e}",
                    GraphMatchWarning,
                )
                ref_atom_orders[ref_i] = np.arange(
                    reference_chains[ref_i].array_length()
                )
                mod_atom_orders[mod_i] = np.arange(pose_chains[mod_i].array_length())
        else:
            ref_atom_orders[ref_i], mod_atom_orders[mod_i] = _find_common_residues(
                reference_chains[ref_i], pose_chains[mod_i]
            )

    # Finally bring chain order and order within chains together
    return (
        _combine_chain_and_atom_orders(
            np.arange(len(reference_chains)), ref_atom_orders, ref_chain_starts
        ),
        _combine_chain_and_atom_orders(
            mod_chain_order, mod_atom_orders, mod_chain_starts
        ),
    )


def _combine_chain_and_atom_orders(
    chain_order: NDArray[np.int_],
    atom_orders: list[NDArray[np.int_]],
    chain_starts: NDArray[np.int_],
) -> NDArray[np.int_]:
    """
    Bring both atom and chain orders together.

    Parameters
    ----------
    chain_order : np.ndarray, shape=(k,), dtype=int
        The order of the chains in terms of indices pointing to a list of chains.
    atom_orders : list of (np.ndarray, shape=(n,), dtype=int), length=k
        The order of the atoms in terms of indices pointing to atoms within each chain.
    chain_starts : np.ndarray, shape=(k+1,), dtype=int
        The start indices of the chains.
        Contains the exclusive stop index of the last chain as last element.

    Returns
    -------
    global_order : np.ndarray, shape=(n,), dtype=int
        The order of the atoms in the global system.
    """
    order_chunks = []
    # Take atom indices for each chain in the determined order
    for chain_i in chain_order:
        atom_indices = np.arange(chain_starts[chain_i], chain_starts[chain_i + 1])
        # Apply reordering within the chain
        atom_indices = atom_indices[atom_orders[chain_i]]
        order_chunks.append(atom_indices)
    return np.concatenate(order_chunks)


def _find_matching_chain_permutation(
    reference_chains: list[struc.AtomArray],
    pose_chains: list[struc.AtomArray],
    min_sequence_identity: float,
) -> tuple[NDArray[np.int_], struc.AffineTransformation, int]:
    """
    Find the permutation of the given chains that minimizes the RMSD between the pose
    and the reference.

    Parameters
    ----------
    reference_chains : list of struc.AtomArray, length=n
        The reference chains.
    pose_chains : list of struc.AtomArray, length=n
        The pose chains.
        Must have the same order as the `reference_chains`,
        i.e. elements at the same index must be equivalent chains.
    min_sequence_identity : float
        The minimum sequence identity between two chains to be considered the same
        entity.

    Returns
    -------
    chain_order : np.array, shape=(n,), dtype=int
        The permutation of the chains that minimizes the RMSD between the pose and the
        reference.
    transform : AffineTransformation
        The transformation that applied on the pose system gave the minimal RMSD.
    anchor_index : int
        The index of the anchor pose chain.
        This index points to the original order, before `chain_order` is applied.
    """
    # Assign reference and pose entity IDs in a single call
    # in order to assign the same ID to corresponding chains between reference and pose
    entity_ids = _assign_entity_ids(
        reference_chains + pose_chains, min_sequence_identity
    )
    # Split the entity IDs again
    reference_entity_ids = entity_ids[: len(reference_chains)]
    pose_entity_ids = entity_ids[len(reference_chains) :]
    if (
        np.bincount(reference_entity_ids, minlength=len(entity_ids))
        != np.bincount(pose_entity_ids, minlength=len(entity_ids))
    ).any():
        raise ValueError("Reference and pose have different entities")

    anchor_index = _choose_anchor_chain(pose_chains, pose_entity_ids)

    reference_centroids = np.array([struc.centroid(c) for c in reference_chains])
    pose_centroids = np.array([struc.centroid(c) for c in pose_chains])

    best_transform = None
    best_rmsd = np.inf
    best_chain_order = None
    # Test all possible chains that represent the same entity against the anchor chain
    for reference_i, reference_chain in enumerate(reference_chains):
        if reference_entity_ids[reference_i] != pose_entity_ids[anchor_index]:
            continue
        else:
            # Superimpose the entire system
            # based on the anchor and chosen reference chain
            transform = _get_superimposition_transform(
                reference_chain, pose_chains[anchor_index]
            )
            chain_order = _find_matching_centroids(
                reference_centroids,
                pose_centroids,
                reference_entity_ids,
                pose_entity_ids,
                transform,
            )
            superimposed_pose_centroids = transform.apply(pose_centroids)
            rmsd = struc.rmsd(
                reference_centroids, superimposed_pose_centroids[chain_order]
            )
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_transform = transform
                best_chain_order = chain_order
    return best_chain_order, best_transform, anchor_index  # type: ignore[return-value]


def _find_matching_centroids(
    reference_centroids: NDArray[np.floating],
    pose_centroids: NDArray[np.floating],
    reference_entity_ids: NDArray[np.int_],
    pose_entity_ids: NDArray[np.int_],
    transform: struc.AffineTransformation,
) -> NDArray[np.int_]:
    """
    Find pairs of chains (each represented by its centroid) between the reference and
    the pose that are closest to each other und the given transformation.

    This functions iteratively chooses pairs with the smallest centroid distance, i.e.
    first the pair with the smallest centroid distance is chosen, then the pair with the
    second smallest centroid distance and so on.

    Parameters
    ----------
    reference_centroids, pose_centroids : np.ndarray, shape=(n,3)
        The centroids of the reference and pose chains.
    reference_entity_ids, pose_entity_ids : np.ndarray, shape=(n,), dtype=int
        The entity IDs of the chains.
        Only centroids of chains with the same entity ID can be matched.
    transform : AffineTransformation
        The transformation that superimposes the pose centroids onto the reference
        centroids.

    Returns
    -------
    pose_order : np.ndarray, shape=(n,)
        The permutation of the pose chains that gives the pairs with the smallest
        distance, i.e. ``pose_order[i] == j`` if the ``i``-th reference chain and
        ``j``-th pose chain are closest to each other.
    """
    pose_centroids = transform.apply(pose_centroids)
    distances = struc.distance(reference_centroids[:, None], pose_centroids[None, :])
    # Different entities must not be matched
    distances[reference_entity_ids[:, None] != pose_entity_ids[None, :]] = np.inf
    pose_order = np.zeros(len(pose_centroids), dtype=int)
    # n chains -> n pairs -> n iterations
    for _ in range(len(pose_centroids)):
        min_distance = np.min(distances)
        min_reference_i, min_pose_i = np.argwhere(distances == min_distance)[0]
        pose_order[min_reference_i] = min_pose_i
        distances[min_reference_i, :] = np.inf
        distances[:, min_pose_i] = np.inf
    return pose_order


def _find_common_residues(
    reference: struc.AtomArray, pose: struc.AtomArray
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Find common residues (and the common atoms) in them in two protein chains.

    Parameters
    ----------
    reference, pose : struc.AtomArray, shape=(n,)
        The reference and pose small molecule, respectively.
        Must have the same atoms (with different coordinates).

    Returns
    -------
    reference_order, pose_order : np.array, shape=(n,), dtype=int
        The atom order that should be applied to the `reference` or `pose`,
        respectively to minimize the RMSD between them.

    Notes
    -----
    Atoms are identified by their element and bond types.
    The element alone is not sufficient, as hydrogen atoms are missing in the
    structures.
    Hence, e.g. a double-bonded oxygen atom would not be distinguishable from a
    hydroxyl group, without the bond types.
    """
    # Shortcut if the structures already match perfectly atom-wise
    if _is_matched(reference, pose, _ANNOTATIONS_FOR_ATOM_MATCHING):
        return np.arange(reference.array_length()), np.arange(reference.array_length())

    reference_sequence = struc.to_sequence(reference)[0][0]
    pose_sequence = struc.to_sequence(pose)[0][0]
    alignment = align.align_optimal(
        reference_sequence,
        pose_sequence,
        _IDENTITY_MATRIX,
        # We get mismatches due to cropping, not due to evolution
        # -> linear gap penalty makes most sense
        gap_penalty=-1,
        max_number=1,
    )[0]
    # Remove gaps -> crop structures to common residues
    alignment.trace = alignment.trace[(alignment.trace != -1).all(axis=1)]

    # Atom masks that are True for atoms in residues that are common in both structures
    reference_mask = _get_mask_from_alignment_trace(reference, alignment.trace[:, 0])
    pose_mask = _get_mask_from_alignment_trace(pose, alignment.trace[:, 1])
    reference_indices = np.arange(reference.array_length())[reference_mask]
    pose_indices = np.arange(pose.array_length())[pose_mask]

    # Within the atoms of aligned residues, select only common atoms
    reference_subset_indices, pose_subset_indices = _find_atom_intersection(
        reference[reference_indices], pose[pose_indices]
    )

    return (
        reference_indices[reference_subset_indices],
        pose_indices[pose_subset_indices],
    )


def _get_mask_from_alignment_trace(
    chain: struc.AtomArray, trace_column: NDArray[np.int_]
) -> NDArray[np.bool_]:
    """
    Get a mask that is True for all atoms, whose residue is contained in the given
    alignment trace column.

    Parameters
    ----------
    chain : AtomArray, shape=(n,)
        The chain to get the mask for.
    trace_column : ndarray, shape=(k,), dtype=int
        The column of the alignment trace for that chain.
        Each index in this trace column points to a residue in the chain.

    Returns
    -------
    mask : ndarray, shape=(n,), dtype=bool
        The mask for the given chain.
    """
    return struc.get_residue_masks(chain, struc.get_residue_starts(chain))[
        trace_column
    ].any(axis=0)


def _find_atom_intersection(
    reference: struc.AtomArray,
    pose: struc.AtomArray,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Find the intersection of two structures, i.e. the set of equivalent atoms.

    Parameters
    ----------
    reference, pose : AtomArray
        The reference and pose chain, respectively.

    Returns
    -------
    common_reference_indices, common_pose_indices : ndarray, shape=(n,), dtype=int
        The reference and pose indices pointing to the common subset of atoms.

    Notes
    -----
    The order is not necessarily canonical, as the atoms are simply sorted based on the
    given annotations.
    The important requirement is that the order is the same for both structures.
    """
    # Shortcut if the structures already match perfectly atom-wise
    if _is_matched(reference, pose, _ANNOTATIONS_FOR_ATOM_MATCHING):
        return np.arange(reference.array_length()), np.arange(reference.array_length())

    # Use continuous residue IDs to enforce that the later reordering does not mix up
    # atoms from different residues
    reference.res_id = struc.create_continuous_res_ids(reference, False)
    pose.res_id = struc.create_continuous_res_ids(pose, False)
    # Implicitly expect that the annotation array dtypes are the same for both
    structured_dtype = np.dtype(
        [
            (name, pose.get_annotation(name).dtype)
            for name in ["res_id"] + _ANNOTATIONS_FOR_ATOM_MATCHING
        ]
    )
    ref_annotations = _annotations_to_structured(reference, structured_dtype)
    mod_annotations = _annotations_to_structured(pose, structured_dtype)
    ref_indices = np.where(np.isin(ref_annotations, mod_annotations))[0]
    mod_indices = np.where(np.isin(mod_annotations, ref_annotations))[0]
    # Atom ordering might not be same -> sort
    mod_indices = mod_indices[np.argsort(mod_annotations[mod_indices])]
    ref_indices = ref_indices[np.argsort(ref_annotations[ref_indices])]

    return ref_indices, mod_indices


def _annotations_to_structured(
    atoms: struc.AtomArray, structured_dtype: np.dtype
) -> NDArray[Any]:
    """
    Convert atom annotations into a single structured `ndarray`.

    Parameters
    ----------
    atoms : AtomArray
        The annotation arrays are taken from this structure.
    structured_dtype : dtype
        The dtype of the structured array to be created.
        The fields of the dtype determine which annotations are taken from `atoms`.
    """
    if structured_dtype.fields is None:
        raise TypeError("dtype must be structured")
    structured = np.zeros(atoms.array_length(), dtype=structured_dtype)
    for field in structured_dtype.fields:
        structured[field] = atoms.get_annotation(field)
    return structured


def _find_optimal_molecule_permutation(
    reference: struc.AtomArray, pose: struc.AtomArray, superimpose: bool
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Find corresponding atoms in small molecules that minimizes the RMSD between the
    pose and the reference.

    Use graph isomorphism on the bond graph to account for symmetries within
    the small molecule.

    Parameters
    ----------
    reference, pose : struc.AtomArray, shape=(n,)
        The reference and pose small molecule, respectively.
        It is expected that they are already superimposed onto each other.
    superimpose : bool
        Whether to superimpose the reference and the reordered pose onto each other
        before calculating the RMSD.
        This is necessary if no other anchor chain determines the optimal
        superimposition.

    Returns
    -------
    reference_order, pose_order : np.array, shape=(n,), dtype=int
        The atom order that should be applied to the `reference` or `pose`,
        respectively to minimize the RMSD between them.

    Notes
    -----
    Atoms are identified by their element only.
    Currently this allows atoms/groups to be matched even if they are not equivalent
    in some scenarios:

    - Configurational diastereomers (enantiomers and cis/trans isomers)
    - Terminal aldehydes and hydroxyl groups
    """
    reference_mol = _to_mol(reference)
    pose_mol = _to_mol(pose)
    mappings = pose_mol.GetSubstructMatches(
        reference_mol, useChirality=True, uniquify=False
    )
    if len(mappings) == 0:
        raise ValueError(
            "No atom mapping found between pose and reference small molecule"
        )

    best_rmsd = np.inf
    best_pose_atom_order = None
    for pose_atom_order in mappings:
        pose_atom_order = np.array(pose_atom_order)
        if len(pose_atom_order) != reference.array_length():
            raise ValueError("Atom mapping does not cover all atoms")
        if superimpose:
            superimposed, _ = struc.superimpose(reference, pose[pose_atom_order])
        else:
            superimposed = pose[pose_atom_order]
        rmsd = struc.rmsd(reference, superimposed)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_pose_atom_order = pose_atom_order
    return np.arange(reference.array_length()), best_pose_atom_order  # type: ignore[return-value]


def _assign_entity_ids(
    chains: list[struc.AtomArray],
    min_sequence_identity: float,
) -> NDArray[np.int_]:
    """
    Assign a unique entity ID to each distinct chain.

    This means that two chains with the same entity ID have sufficient sequence
    identity or in case of small molecules have the same ``res_name``.

    Parameters
    ----------
    chains : list of struc.AtomArray, length=n
        The chains to assign entity IDs to.
    min_sequence_identity : float
        The minimum sequence identity between two chains to be considered the same
        entity.

    Returns
    -------
    entity_ids : np.ndarray, shape=(n,), dtype=int
        The entity IDs.
    """
    sequences = [
        struc.to_sequence(chain)[0][0] if not is_small_molecule(chain) else None
        for chain in chains
    ]

    current_entity_id = 0
    entity_ids: list[int] = []
    for i, (chain, sequence) in enumerate(zip(chains, sequences)):
        for j in range(i):
            if sequence is None:
                # Match small molecules by residue name
                if chain.res_name[0] == chains[j].res_name[0]:
                    entity_ids.append(entity_ids[j])
                    break
            else:
                # Match protein chains by sequence identity
                alignment = align.align_optimal(
                    sequence,
                    sequences[j],
                    _IDENTITY_MATRIX,
                    # We get mismatches due to experimental artifacts, not evolution
                    # -> linear gap penalty makes most sense
                    gap_penalty=-1,
                    max_number=1,
                )[0]
                if (
                    align.get_sequence_identity(alignment, mode="all")
                    >= min_sequence_identity
                ):
                    entity_ids.append(entity_ids[j])
                    break
        else:
            # No match found to a chain that already has an entity ID -> assign new ID
            entity_ids.append(current_entity_id)
            current_entity_id += 1

    return np.array(entity_ids)


def _choose_anchor_chain(
    chains: list[struc.AtomArray], entity_ids: NDArray[np.int_]
) -> int:
    """
    Choose the anchor chain for the RMSD calculation.

    The most preferable chain is the one with the least multiplicity and the longest
    sequence.

    Parameters
    ----------
    chains : list of struc.AtomArray, length=n
        The chains to choose from.
    entity_ids : ndarray, shape=(n,), dtype=int
        The entity IDs of the chains.
        Used to determine the multiplicity of each chain.

    Returns
    -------
    anchor_chain : int
        The index of the anchor chain.
    """
    protein_chain_indices = np.where(
        [not is_small_molecule(chain) for chain in chains]
    )[0]
    if len(protein_chain_indices) == 0:
        # No protein chains -> Simply use the first small molecule as anchor
        return 0

    protein_entity_ids = entity_ids[protein_chain_indices]
    multiplicities_of_entity_ids = np.bincount(protein_entity_ids)
    multiplicities = multiplicities_of_entity_ids[protein_entity_ids]
    least_multiplicity_indices = np.where(multiplicities == np.min(multiplicities))[0]
    # Use the sequence length as tiebreaker
    sequence_lengths = np.array([len(chains[i]) for i in protein_chain_indices])
    # Only consider the lengths of the preselected chains
    largest_length = np.max(sequence_lengths[least_multiplicity_indices])
    largest_length_indices = np.where(sequence_lengths == largest_length)[0]
    best_anchors = np.intersect1d(least_multiplicity_indices, largest_length_indices)
    return protein_chain_indices[best_anchors[0]]


def _get_superimposition_transform(
    reference_chain: struc.AtomArray, pose_chain: struc.AtomArray
) -> struc.AffineTransformation:
    """
    Get the transformation (translation and rotation) that superimposes the pose chain
    onto the reference chain.

    Parameters
    ----------
    reference_chain, pose_chain : AtomArray
        The chains to superimpose.

    Returns
    -------
    transform : AffineTransformation
        The transformation that superimposes the pose chain onto the reference chain.
    """
    if is_small_molecule(reference_chain):
        if reference_chain.array_length() == pose_chain.array_length():
            _, transform = struc.superimpose(reference_chain, pose_chain)
        else:
            # The small molecules have different lengths -> difficult superimposition
            # -> simply get an identity transformation
            # This case can only happen in small molecule-only systems anyway
            transform = struc.AffineTransformation(
                center_translation=np.zeros(3),
                rotation=np.eye(3),
                target_translation=np.zeros(3),
            )
    else:
        _, transform, _, _ = struc.superimpose_homologs(
            reference_chain,
            pose_chain,
            _IDENTITY_MATRIX,
            gap_penalty=-1,
            min_anchors=1,
            # No outlier removal
            max_iterations=1,
        )
    return transform


def _is_matched(
    reference: struc.AtomArray,
    pose: struc.AtomArray,
    annotation_names: list[str],
) -> bool:
    """
    Check if the given annotations are the same in both structures.

    Parameters
    ----------
    pose, reference : AtomArray
        The pose and reference structure to be compared, respectively.
    annotation_names : list of str
        The names of the annotations to be compared.

    Returns
    -------
    matched : bool
        True, if the annotations are the same in both structures.
    """
    if reference.array_length() != pose.array_length():
        return False
    for annot_name in annotation_names:
        if not (
            reference.get_annotation(annot_name) == pose.get_annotation(annot_name)
        ).all():
            return False
    return True


def _to_mol(molecule: struc.AtomArray) -> Mol:
    """
    Create a RDKit :class:`Mol` from the given structure and prepare it for usage in
    atom matching.

    Parameters
    ----------
    molecule : struc.AtomArray
        The molecule to convert.

    Returns
    -------
    mol : Mol
        The RDKit molecule.
    """
    mol = rdkit_interface.to_mol(molecule)
    # Make RDKit distinguish stereoisomers when matching atoms
    AssignStereochemistryFrom3D(mol)
    # Make conjugated terminal groups symmetric (e.g. carboxyl oxygen atoms)
    SanitizeMol(mol, SanitizeFlags.SANITIZE_SETCONJUGATION)
    for bond in mol.GetBonds():
        if bond.GetIsConjugated() and bond.GetBondType() in [
            BondType.SINGLE,
            BondType.DOUBLE,
        ]:
            bond.SetBondType(BondType.ONEANDAHALF)
    return mol
