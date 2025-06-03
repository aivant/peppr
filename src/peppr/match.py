__all__ = [
    "GraphMatchWarning",
    "UnmappableEntityError",
    "find_optimal_match",
    "find_all_matches",
]


import itertools
import warnings
from typing import Any, Iterator
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


class UnmappableEntityError(Exception):
    """
    This exceptions is raised, if the reference and pose structure contain
    entities that cannot be mapped to each other.
    """

    pass


def find_optimal_match(
    reference: struc.AtomArray,
    pose: struc.AtomArray,
    min_sequence_identity: float = 0.95,
    use_heuristic: bool = True,
    max_matches: int | None = None,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Find the atom indices for the given reference and pose structure that brings these
    structure into a corresponding order that minimizes the RMSD between them.

    Parameters
    ----------
    reference : AtomArray, shape=(p,)
        The reference structure.
    pose : AtomArray, shape=(q,)
        The pose structure.
    min_sequence_identity : float
        The minimum sequence identity between two chains to be considered the same
        entity.
    use_heuristic : bool or int
        Whether to employ a fast heuristic [1]_ to find the optimal chain permutation.
        This heuristic represents each chain by its centroid, i.e. instead of
        exhaustively superimposing all atoms for each permutation, only the centroids
        are superimposed and the closest match between the reference and pose is
        selected.
    max_matches : int, optional
        The maximum number of atom mappings to try, if the `use_heuristic` is set to
        ``False``.

    Returns
    -------
    reference_indices : np.array, shape=(n,), dtype=int
        The atom indices that should be applied to `reference`.
    pose_indices : np.array, shape=(n,), dtype=int
        The atom indices that should be applied to `pose`.

    Notes
    -----
    Note that the heuristic used by default is much faster compared to the
    exhaustive approach:
    Especially for larger complexes with many homomers or small molecule copies,
    the number of possible mappings combinatorially explodes.
    However, the heuristic might not find the optimal permutation for all cases,
    especially in poses that only remotely resemble the reference.

    References
    ----------
    .. [1] *Protein complex prediction with AlphaFold-Multimer*, Section 7.3, https://doi.org/10.1101/2021.10.04.463034
    """
    reference_chains = list(struc.chain_iter(reference))
    pose_chains = list(struc.chain_iter(pose))
    if len(reference_chains) == 1:
        # Only one chain -> no need to match
        if len(pose_chains) != 1:
            raise UnmappableEntityError("Reference and pose have different entities")
        return _create_indices_from_chain_order(
            reference_chains,
            pose_chains,
            np.array([0]),
            np.array([0]),
            superimpose=True,
        )

    if use_heuristic:
        return _find_optimal_match_fast(
            reference_chains, pose_chains, min_sequence_identity
        )
    else:
        return _find_optimal_match_precise(
            reference_chains, pose_chains, min_sequence_identity, max_matches
        )


def find_all_matches(
    reference: struc.AtomArray,
    pose: struc.AtomArray,
    min_sequence_identity: float = 0.95,
) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """
    Find all possible atom mappings between the reference and the pose.

    Each mappings gives corresponding atoms between the reference and the pose.

    Parameters
    ----------
    reference : AtomArray, shape=(p,)
        The reference structure.
    pose : AtomArray, shape=(q,)
        The pose structure.
    min_sequence_identity : float
        The minimum sequence identity between two chains to be considered the same
        entity.

    Yields
    ------
    list of tuple (np.ndarray, shape=(n,), dtype=int)
        Atom indices that, when applied to the reference and pose,
        respectively, yield the corresponding atoms.

    Notes
    -----
    This functions tries all chain mappings of chain that are the same entity and
    within each small molecules tries all proper molecule permutations.
    For larger homomers this can quickly lead to a combinatorial explosion.
    """
    reference_chains = list(struc.chain_iter(reference))
    pose_chains = list(struc.chain_iter(pose))
    for m in _all_global_mappings(reference_chains, pose_chains, min_sequence_identity):
        yield m


def _find_optimal_match_fast(
    reference_chains: list[struc.AtomArray],
    pose_chains: list[struc.AtomArray],
    min_sequence_identity: float = 0.95,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Find the optimal atom order for each pose that minimizes the centroid RMSD to the
    reference.

    Parameters
    ----------
    reference_chains, pose_chains : list of AtomArray
        The reference and pose structure, separated into chains.
    min_sequence_identity : float
        The minimum sequence identity between two chains to be considered the same
        entity.

    Returns
    -------
    reference_order, pose_order : np.array, shape=(n,), dtype=int
        The atom order that should be applied to `reference` and pose, respectively.
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
        raise UnmappableEntityError("Reference and pose have different entities")

    anchor_index = _choose_anchor_chain(pose_chains, pose_entity_ids)

    # Find corresponding chains by identifying the chain permutation minimizing the centroid RMSD
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

    pose_chains = [best_transform.apply(chain) for chain in pose_chains]  # type: ignore[union-attr]
    return _create_indices_from_chain_order(
        reference_chains,
        pose_chains,
        np.arange(len(reference_chains)),
        best_chain_order,
        # Superimposition is defined by centroids
        superimpose=False,
    )


def _find_optimal_match_precise(
    reference_chains: list[struc.AtomArray],
    pose_chains: list[struc.AtomArray],
    min_sequence_identity: float = 0.95,
    max_matches: int | None = None,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Find the optimal atom order for each pose that minimizes the all-atom RMSD to the
    reference.

    Parameters
    ----------
    reference_chains, pose_chains : list of AtomArray
        The reference and pose structure, separated into chains.
    min_sequence_identity : float
        The minimum sequence identity between two chains to be considered the same
        entity.
    max_matches : int, optional
        The maximum number of mappings to try.

    Returns
    -------
    reference_order, pose_order : np.array, shape=(n,), dtype=int
        The atom order that should be applied to `reference` and pose, respectively.
    """
    if max_matches is not None and max_matches < 1:
        raise ValueError("Maximum number of mappings must be at least 1")

    # The reassembled structures are required below
    # for global superimposition and RMSD computation
    reference = struc.concatenate(reference_chains)
    pose = struc.concatenate(pose_chains)

    best_rmsd = np.inf
    best_indices: tuple[NDArray[np.int_], NDArray[np.int_]] = None  # type: ignore[assignment]
    for it, (ref_indices, pose_indices) in enumerate(
        _all_global_mappings(reference_chains, pose_chains, min_sequence_identity)
    ):
        if max_matches is not None and it >= max_matches:
            break
        matched_reference_coord = reference.coord[ref_indices]
        matched_pose_coord, _ = struc.superimpose(
            matched_reference_coord, pose.coord[pose_indices]
        )
        rmsd = struc.rmsd(matched_reference_coord, matched_pose_coord)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_indices = ref_indices, pose_indices

    return best_indices


def _all_global_mappings(
    reference_chains: list[struc.AtomArray],
    pose_chains: list[struc.AtomArray],
    min_sequence_identity: float = 0.95,
) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """
    Find all possible atom mappings between the reference and the pose.

    Each mappings gives corresponding atoms between the reference and the pose.

    Parameters
    ----------
    reference_chains, pose_chains : list of AtomArray
        The reference and pose chains, respectively.
    min_sequence_identity : float
        The minimum sequence identity between two chains to be considered the same
        entity.

    Yields
    ------
    ref_atom_indices, pose_atom_indices : list of np.ndarray, shape=(n,), dtype=int
        The global atom indices that, when applied to the entire reference and pose,
        respectively, yield the corresponding atoms.

    Notes
    -----
    This functions tries all chain mappings of chain that are the same entity and
    within each small molecules tries all proper molecule permutations.
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
        raise UnmappableEntityError("Reference and pose have different entities")

    ref_chain_order = np.arange(len(reference_chains), dtype=int)
    for pose_chain_order in itertools.permutations(range(len(pose_chains))):
        pose_chain_order = np.array(pose_chain_order)  # type: ignore[assignment]
        if np.any(
            reference_entity_ids[ref_chain_order] != pose_entity_ids[pose_chain_order]
        ):
            # This permutation would map different entities to each other
            continue
        # All possible intra-chain atom mappings for the current chain mapping
        # In case of symmetric small molecules, multiple mappings are possible
        all_intra_chain_mappings: list[list[tuple[np.ndarray, np.ndarray]]] = []
        for ref_chain_i, pose_chain_i in zip(ref_chain_order, pose_chain_order):
            if is_small_molecule(reference_chains[ref_chain_i]):
                try:
                    pose_mappings = _molecule_mappings(
                        reference_chains[ref_chain_i], pose_chains[pose_chain_i]
                    )
                    all_mapping_possibilities = []
                    for pose_mapping in pose_mappings:
                        all_mapping_possibilities.append(
                            (
                                np.arange(reference_chains[ref_chain_i].array_length()),
                                pose_mapping,
                            )
                        )
                    all_intra_chain_mappings.append(all_mapping_possibilities)
                except Exception as e:
                    warnings.warn(
                        f"RDKit failed atom matching: {e}",
                        GraphMatchWarning,
                    )
                    # Default to trivial mapping
                    all_intra_chain_mappings.append(
                        [
                            (
                                np.arange(reference_chains[ref_chain_i].array_length()),
                                np.arange(pose_chains[pose_chain_i].array_length()),
                            )
                        ]
                    )
            else:
                # For polymers there is only one mapping
                all_intra_chain_mappings.append(
                    [
                        _find_common_residues(
                            reference_chains[ref_chain_i], pose_chains[pose_chain_i]
                        )
                    ]
                )

        # Create Cartesian product of intra-chain mappings over all chains
        for intra_chain_mappings in itertools.product(*all_intra_chain_mappings):
            ref_atom_orders = [mapping[0] for mapping in intra_chain_mappings]
            pose_atom_orders = [mapping[1] for mapping in intra_chain_mappings]
            # The atom orders are still in the permuted chain order
            # -> reorder them to the original chain order
            ref_atom_orders = [ref_atom_orders[i] for i in np.argsort(ref_chain_order)]
            pose_atom_orders = [
                pose_atom_orders[i] for i in np.argsort(pose_chain_order)
            ]
            yield (
                _combine_inter_and_intra_chain_orders(
                    ref_chain_order,
                    ref_atom_orders,
                    [chain.array_length() for chain in reference_chains],
                ),
                _combine_inter_and_intra_chain_orders(
                    pose_chain_order,
                    pose_atom_orders,
                    [chain.array_length() for chain in pose_chains],
                ),
            )


def _create_indices_from_chain_order(
    reference_chains: list[struc.AtomArray],
    pose_chains: list[struc.AtomArray],
    reference_chain_order: NDArray[np.int_],
    pose_chain_order: NDArray[np.int_],
    superimpose: bool = False,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Given the order of corresponding chains between the reference and the pose,
    find the corresponding atom order within each chain and create atom indices
    that point to the entire structure.

    Parameters
    ----------
    reference_chains, pose_chains : list of AtomArray
        The reference and pose chains, respectively.
    reference_chain_order, pose_chain_order : np.ndarray, shape=(n,), dtype=int
        The indices when applied to `reference_chains` and `pose_chains`, respectively,
        yield the corresponding chains.
    superimpose : bool, optional
        Whether to superimpose the corresponding small molecules onto each other
        before finding the optimal permutation.

    Returns
    -------
    ref_atom_indices, pose_atom_indices : list of np.ndarray, shape=(n,), dtype=int
        The global atom indices that, when applied to the entire reference and pose,
        respectively, yield the corresponding atoms.
    """
    ref_atom_orders: list[np.ndarray | None] = [None] * len(reference_chains)
    pose_atom_orders: list[np.ndarray | None] = [None] * len(pose_chains)
    for ref_i, pose_i in zip(reference_chain_order, pose_chain_order):
        if is_small_molecule(reference_chains[ref_i]):
            try:
                ref_atom_orders[ref_i], pose_atom_orders[pose_i] = (
                    _find_optimal_molecule_permutation(
                        reference_chains[ref_i],
                        pose_chains[pose_i],
                        superimpose=superimpose,
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
                pose_atom_orders[pose_i] = np.arange(pose_chains[pose_i].array_length())
        else:
            ref_atom_orders[ref_i], pose_atom_orders[pose_i] = _find_common_residues(
                reference_chains[ref_i], pose_chains[pose_i]
            )

    # Finally bring chain order and atom order within chains together
    return (
        _combine_inter_and_intra_chain_orders(
            reference_chain_order,
            ref_atom_orders,
            [chain.array_length() for chain in reference_chains],
        ),
        _combine_inter_and_intra_chain_orders(
            pose_chain_order,
            pose_atom_orders,
            [chain.array_length() for chain in pose_chains],
        ),
    )


def _combine_inter_and_intra_chain_orders(
    chain_order: NDArray[np.int_],
    atom_orders: list[NDArray[np.int_]],
    chain_lengths: NDArray[np.int_],
) -> NDArray[np.int_]:
    """
    Bring both, chain indices and atom indices within each chain, together.

    Parameters
    ----------
    chain_order : np.ndarray, shape=(k,), dtype=int
        The order of the chains in terms of indices pointing to a list of chains.
    atom_orders : list of (np.ndarray, shape=(n,), dtype=int), length=k
        The order of the atoms in terms of indices pointing to atoms within each chain.
    chain_lengths : np.ndarray, shape=(k,), dtype=int
        The number of atoms in each chain.

    Returns
    -------
    global_order : np.ndarray, shape=(n,), dtype=int
        The order of the atoms in the global system.
    """
    chain_starts = np.concatenate(([0], np.cumsum(chain_lengths)))
    order_chunks = []
    # Take atom indices for each chain in the determined order
    for chain_i in chain_order:
        atom_indices = np.arange(chain_starts[chain_i], chain_starts[chain_i + 1])
        # Apply reordering within the chain
        atom_indices = atom_indices[atom_orders[chain_i]]
        order_chunks.append(atom_indices)
    return np.concatenate(order_chunks)


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
    Find common residues (and the common atoms) in two protein chains.

    Parameters
    ----------
    reference, pose : struc.AtomArray, shape=(n,)
        The reference and pose polymer chain whose common atoms are to be found.

    Returns
    -------
    reference_indices, pose_indices : np.array, shape=(n,), dtype=int
        The indices that when applied to the `reference` or `pose`, respectively,
        yield the corresponding atoms.

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
        terminal_penalty=False,
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
    pose_annotations = _annotations_to_structured(pose, structured_dtype)
    ref_indices = np.where(np.isin(ref_annotations, pose_annotations))[0]
    pose_indices = np.where(np.isin(pose_annotations, ref_annotations))[0]
    # Atom ordering might not be same -> sort
    pose_indices = pose_indices[np.argsort(pose_annotations[pose_indices])]
    ref_indices = ref_indices[np.argsort(ref_annotations[ref_indices])]

    return ref_indices, pose_indices


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
    superimpose : bool
        Whether to superimpose the reference and the reordered pose onto each other
        before calculating the RMSD.
        Only necessary if no other means has already determined the superimposition.
        If ``False``, it is expected that they are already superimposed onto each other.

    Returns
    -------
    reference_order, pose_order : np.array, shape=(n,), dtype=int
        The indices that when applied to the `reference` or `pose`, respectively,
        yield the corresponding atoms.
    """
    best_rmsd = np.inf
    best_pose_atom_order = None
    for pose_atom_order in _molecule_mappings(reference, pose):
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


def _molecule_mappings(
    reference: struc.AtomArray, pose: struc.AtomArray
) -> list[NDArray[np.int_]]:
    """
    Find corresponding atoms in small molecules.

    Use graph isomorphism on the bond graph to account for symmetries within
    the small molecule.

    Parameters
    ----------
    reference, pose : struc.AtomArray, shape=(n,)
        The reference and pose small molecule, respectively.
        It is expected that they are already superimposed onto each other.

    Returns
    -------
    list of np.ndarray, shape=(n,), dtype=int
        All mappings of `pose` that maps its atoms to the `reference` with
        respect to elements and the bond graph.
    """
    reference_mol = _to_mol(reference)
    pose_mol = _to_mol(pose)
    mappings = pose_mol.GetSubstructMatches(
        reference_mol, useChirality=True, uniquify=False
    )
    if len(mappings) == 0:
        # The coordinates may be off, so try matching without chirality
        mappings = pose_mol.GetSubstructMatches(
            reference_mol, useChirality=False, uniquify=False
        )
    if len(mappings) == 0:
        # If still no match is found, conformation is not the problem,
        # but the bond graph is different -> correct match is impossible
        raise ValueError(
            "No atom mapping found between pose and reference small molecule"
        )
    # Convert tuples to proper index arrays
    return [np.array(mapping) for mapping in mappings]


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
            if (sequence is None and sequences[j] is not None) or (
                sequence is not None and sequences[j] is None
            ):
                # Cannot match small molecules to polymer chains
                continue
            elif sequence is None:
                # Match small molecules by residue name
                if chain.res_name[0] == chains[j].res_name[0]:
                    entity_ids.append(entity_ids[j])
                    break
            else:
                # Match polymer chains by sequence identity
                alignment = align.align_optimal(
                    sequence,
                    sequences[j],
                    _IDENTITY_MATRIX,
                    # We get mismatches due to experimental artifacts, not evolution
                    # -> linear gap penalty makes most sense
                    gap_penalty=-1,
                    terminal_penalty=False,
                    max_number=1,
                )[0]
                if (
                    align.get_sequence_identity(alignment, mode="shortest")
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
    SanitizeMol(
        mol,
        SanitizeFlags.SANITIZE_SETCONJUGATION | SanitizeFlags.SANITIZE_SETAROMATICITY,
    )
    # Make conjugated terminal groups truly symmetric (e.g. carboxyl oxygen atoms)
    for bond in mol.GetBonds():
        if bond.GetIsConjugated() and bond.GetBondType() in [
            BondType.SINGLE,
            BondType.DOUBLE,
        ]:
            bond.SetBondType(BondType.ONEANDAHALF)
    return mol
