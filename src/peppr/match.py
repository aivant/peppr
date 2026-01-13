__all__ = [
    "GraphMatchWarning",
    "UnmappableEntityError",
    "StructureMismatchError",
    "filter_matched",
    "find_optimal_match",
    "find_all_matches",
    "find_matching_centroids",
]

import itertools
import warnings
from collections import Counter
from collections.abc import Callable
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
from peppr.common import MoleculeType

# To match atoms between pose and reference chain,
# the residue and atom name are sufficient to unambiguously identify an atom
_ANNOTATIONS_FOR_ATOM_MATCHING = ["res_name", "atom_name"]
_IDENTITY_MATRIX = {
    molecule_type: align.SubstitutionMatrix(
        alphabet,
        alphabet,
        np.eye(len(alphabet), dtype=np.int32),
    )
    for molecule_type, alphabet in (
        (MoleculeType.PROTEIN, seq.ProteinSequence.alphabet),
        (MoleculeType.NUCLEIC_ACID, seq.NucleotideSequence.alphabet_amb),
    )
}
_PADDING = -1
_UNMAPPABLE_ENTITY_ID = -1


class GraphMatchWarning(UserWarning):
    """
    This warning is raised if the RDKit based molecule matching fails.
    In this case small molecule reordering is skipped.
    """

    pass


class UnmappableEntityError(Exception):
    """
    This exception is raised if the reference and pose structure contain
    entities that cannot be mapped to each other.
    """

    pass


class StructureMismatchError(Exception):
    """
    This exception is raised if the reference and pose structure filtered to
    ``matched`` atoms do not actually match.
    This indicates some issue in the matching process.
    """

    pass


def filter_matched(
    reference: struc.AtomArray | struc.AtomArrayStack,
    pose: struc.AtomArray | struc.AtomArrayStack,
    prefilter: Callable[[struc.AtomArray], NDArray[np.bool_]] | None = None,
) -> tuple[
    struc.AtomArray | struc.AtomArrayStack, struc.AtomArray | struc.AtomArrayStack
]:
    """
    Filter the matched atoms from the reference and pose, i.e.
    where their ``matched`` annotation is ``True``.

    Parameters
    ----------
    reference, pose : AtomArray
        The structures to filter.
        If they have a ``matched`` annotation, the atoms where ``matched`` is ``True``
        are kept.
        Otherwise, all atoms are kept.
    prefilter : Callable[AtomArray -> ndarray, dtype=bool], optional
        For convenience, an additional filter function can be applied to both, the
        reference and pose, before additionally filtering the matched atoms.

    Returns
    -------
    matched_reference, matched_pose : AtomArray or AtomArrayStack
        The filtered reference and pose.

    Raises
    ------
    StructureMismatchError
        If the matched structures do not have corresponding atoms, indicating an issue
        in the prior matching process.
    """
    if "matched" in reference.get_annotation_categories():
        ref_mask = reference.matched
    else:
        ref_mask = np.full(reference.array_length(), True)
    if "matched" in pose.get_annotation_categories():
        pose_mask = pose.matched
    else:
        pose_mask = np.full(pose.array_length(), True)
    if prefilter is not None:
        ref_mask = ref_mask & prefilter(reference)
        pose_mask = pose_mask & prefilter(pose)
    reference = reference[..., ref_mask]
    pose = pose[..., pose_mask]
    if reference.array_length() != pose.array_length():
        raise StructureMismatchError(
            f"Filtered reference has {reference.array_length()} atoms, "
            f"but filtered pose has {pose.array_length()} atoms"
        )
    if np.any(np.char.upper(reference.element) != np.char.upper(pose.element)):
        raise StructureMismatchError(
            "Filtered reference and pose have different chemical elements "
            "mapped to each other"
        )
    return reference, pose


def find_optimal_match(
    reference: struc.AtomArray,
    pose: struc.AtomArray,
    min_sequence_identity: float = 0.95,
    use_heuristic: bool = True,
    max_matches: int | None = None,
    allow_unmatched_entities: bool = False,
    use_entity_annotation: bool = False,
    use_structure_match: bool = False,
) -> tuple[struc.AtomArray, struc.AtomArray]:
    """
    Match the atoms from the given reference and pose structure so that the RMSD between
    them is minimized.

    'Matching' has two effects here:

    - Chains and atoms within each residue *that have a counterpart* in the respective
      other structure, are reordered if necessary so that they are in the same order.
    - A ``matched`` annotation is added, which is ``False`` for all atoms, that
      *do not have a counterpart*.

    Parameters
    ----------
    reference : AtomArray, shape=(p,)
        The reference structure.
    pose : AtomArray, shape=(q,)
        The pose structure.
    min_sequence_identity : float, optional
        The minimum sequence identity between two chains to be considered the same
        entity.
    use_heuristic : bool or int, optional
        Whether to employ a fast heuristic [1]_ to find the optimal chain permutation.
        This heuristic represents each chain by its centroid, i.e. instead of
        exhaustively superimposing all atoms for each permutation, only the centroids
        are superimposed and the closest match between the reference and pose is
        selected.
    max_matches : int, optional
        The maximum number of atom mappings to try, if the `use_heuristic` is set to
        ``False``.
    allow_unmatched_entities : bool, optional
        If set to ``True``, allow entire entities to be unmatched.
        This is useful if a pose is compared to a reference which may contain different
        molecules.
    use_entity_annotation : bool, optional
        If set to ``True``, use the ``entity_id`` annotation to determine which chains
        are the same entity and therefore are mappable to each other.
        By default, the entity is determined from sequence identity for polymers and
        residue name for small molecules.
    use_structure_match : bool, optional
        If set to ``True``, use structure matching, i.e. isomorphism on the bond graph,
        to determine which small molecules are the same entity.
        Otherwise, match small molecules by residue name.
        Note that the structure match requires more computation time.

    Returns
    -------
    matched_reference, matched_pose : AtomArray, shape=(p,) or (q,)
        The input atoms, where the chains and atoms within each residue are brought into
        the corresponding order.
        Atoms that are matched between the reference and the pose are annotated with
        ``matched=True``.
        All other atoms are annotated with ``matched=False``.
        This means indexing both structures with ``matched`` as boolean mask will return
        structures with the same number of atoms.

    Notes
    -----
    Atoms that are not matched (``matched=False``), are positioned in the reordered
    return value as follows:

    - Unmatched chains are appended to the end.
    - Unmatched residues within a matched chain are kept at their original sequence
      position.
    - Unmatched atoms within a matched residue are kept at their original position.

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
    if len(reference_chains) == 0:
        raise UnmappableEntityError("Reference is empty")
    if len(pose_chains) == 0:
        raise UnmappableEntityError("Pose is empty")

    if use_heuristic:
        return _find_optimal_match_fast(
            reference_chains,
            pose_chains,
            min_sequence_identity,
            allow_unmatched_entities,
            use_entity_annotation,
            use_structure_match,
        )
    else:
        return _find_optimal_match_precise(
            reference_chains,
            pose_chains,
            min_sequence_identity,
            max_matches,
            allow_unmatched_entities,
            use_entity_annotation,
            use_structure_match,
        )


def find_all_matches(
    reference: struc.AtomArray,
    pose: struc.AtomArray,
    min_sequence_identity: float = 0.95,
    allow_unmatched_entities: bool = False,
    use_entity_annotation: bool = False,
    use_structure_match: bool = False,
) -> Iterator[tuple[struc.AtomArray, struc.AtomArray]]:
    """
    Find all possible atom mappings between the reference and the pose.

    Each mappings gives corresponding atoms between the reference and the pose.

    Parameters
    ----------
    reference : AtomArray, shape=(p,)
        The reference structure.
    pose : AtomArray, shape=(q,)
        The pose structure.
    min_sequence_identity : float, optional
        The minimum sequence identity between two chains to be considered the same
        entity.
    allow_unmatched_entities : bool, optional
        If set to ``True``, allow entire entities to be unmatched.
        This is useful if a pose is compared to a reference which may contain different
        molecules.
    use_entity_annotation : bool, optional
        If set to ``True``, use the ``entity_id`` annotation to determine which chains
        are the same entity and therefore are mappable to each other.
        By default, the entity is determined from sequence identity for polymers and
        residue name for small molecules.
    use_structure_match : bool, optional
        If set to ``True``, use structure matching, i.e. isomorphism on the bond graph,
        to determine which small molecules are the same entity.
        Otherwise, match small molecules by residue name.
        Note that the structure match requires more computation time.

    Yields
    ------
    list of tuple (AtomArray, shape=(p,), AtomArray, shape=(q,))
        The input atoms, where the chains and atoms within each chain are brought into
        the corresponding order.
        Atoms that are matched between the reference and the pose are annotated with
        ``matched=True``.
        All other atoms are annotated with ``matched=False``.
        This means indexing both structures with ``matched`` as boolean mask will return
        structures with the same number of atoms.

    See Also
    --------
    find_optimal_match : More detailed information.

    Notes
    -----
    This functions tries all chain mappings of chain that are the same entity and
    within each small molecules tries all proper molecule permutations.
    For larger homomers this can quickly lead to a combinatorial explosion.
    """
    reference_chains = list(struc.chain_iter(reference))
    pose_chains = list(struc.chain_iter(pose))
    for m in _all_global_mappings(
        reference_chains,
        pose_chains,
        min_sequence_identity,
        allow_unmatched_entities,
        use_entity_annotation,
        use_structure_match,
    ):
        yield m


def find_matching_centroids(
    reference_centroids: NDArray[np.floating],
    pose_centroids: NDArray[np.floating],
    reference_entity_ids: NDArray[np.int_] | None = None,
    pose_entity_ids: NDArray[np.int_] | None = None,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Greedily find pairs of chains (each represented by its centroid) between the
    reference and the pose that are closest to each other.

    This functions iteratively chooses pairs with the smallest centroid distance, i.e.
    first the pair with the smallest centroid distance is chosen, then the pair with the
    second smallest centroid distance and so on.

    Parameters
    ----------
    reference_centroids, pose_centroids : np.ndarray, shape=(n,3)
        The centroids of the reference and pose chains.
    reference_entity_ids, pose_entity_ids : np.ndarray, shape=(n,), dtype=int, optional
        The entity IDs of the chains.
        Only centroids of chains with the same entity ID can be matched.
        By default, all can be matched to each other.

    Returns
    -------
    reference_chain_indices, pose_chain_indices : np.ndarray, shape=(n,)
        Indices to corresponding chains in the reference and pose that gives the pairs
        with the smallest distance to each other.
    """
    if reference_entity_ids is None or pose_entity_ids is None:
        # Assign the same entity id to all chains
        reference_entity_ids = np.zeros(len(reference_centroids), dtype=int)
        pose_entity_ids = np.zeros(len(pose_centroids), dtype=int)

    distances = struc.distance(reference_centroids[:, None], pose_centroids[None, :])
    # Different entities must not be matched
    distances[reference_entity_ids[:, None] != pose_entity_ids[None, :]] = np.inf
    # Unmappable entities must not be matched
    distances[reference_entity_ids == _UNMAPPABLE_ENTITY_ID, :] = np.inf
    distances[:, pose_entity_ids == _UNMAPPABLE_ENTITY_ID] = np.inf

    reference_chain_indices = []
    pose_chain_indices = []
    while True:
        min_distance = np.min(distances)
        if np.isinf(min_distance):
            # No chains can be matched to each other anymore
            break
        min_reference_i, min_pose_i = np.argwhere(distances == min_distance)[0]
        reference_chain_indices.append(min_reference_i)
        pose_chain_indices.append(min_pose_i)
        distances[min_reference_i, :] = np.inf
        distances[:, min_pose_i] = np.inf
    reference_chain_indices = np.array(reference_chain_indices)
    pose_chain_indices = np.array(pose_chain_indices)
    # Try to keep the reference chains in the original order if possible
    order = np.argsort(reference_chain_indices)
    reference_chain_indices = reference_chain_indices[order]
    pose_chain_indices = pose_chain_indices[order]
    return reference_chain_indices, pose_chain_indices


def _find_optimal_match_fast(
    reference_chains: list[struc.AtomArray],
    pose_chains: list[struc.AtomArray],
    min_sequence_identity: float,
    allow_unmatched_entities: bool,
    use_entity_annotation: bool,
    use_structure_match: bool,
) -> tuple[struc.AtomArray, struc.AtomArray]:
    """
    Find matching atoms that minimize the centroid RMSD between the pose and the
    reference.

    Parameters
    ----------
    reference_chains, pose_chains : list of AtomArray
        The reference and pose structure, separated into chains.
    min_sequence_identity : float, optional
        The minimum sequence identity between two chains to be considered the same
        entity.
    allow_unmatched_entities : bool, optional
        If set to ``True``, allow entire entities to be unmatched.
        This is useful if a pose is compared to a reference which may contain different
        molecules.
    use_entity_annotation : bool, optional
        If set to ``True``, use the ``entity_id`` annotation to determine which chains
        are the same entity and therefore are mappable to each other.
    use_structure_match : bool, optional
        If set to ``True``, use structure matching, i.e. isomorphism on the bond graph,
        to determine which small molecules are the same entity.
        Otherwise, match small molecules by residue name.
        Note that the structure match requires more computation time.

    Returns
    -------
    matched_reference, matched_pose : AtomArray
        The input atoms, where the chains and atoms within each chain are brought into
        the corresponding order.
    """
    reference_entity_ids, pose_entity_ids = _assign_entity_ids(
        reference_chains,
        pose_chains,
        min_sequence_identity,
        allow_unmatched_entities,
        use_entity_annotation,
        use_structure_match,
    )

    # Find corresponding chains by identifying the chain permutation that minimizes
    # the centroid RMSD
    reference_centroids = np.array([struc.centroid(c) for c in reference_chains])
    pose_centroids = np.array([struc.centroid(c) for c in pose_chains])
    best_transform = None
    best_rmsd = np.inf
    best_reference_indices = None
    best_pose_indices = None
    # Test all possible chains that represent the same entity against the anchor chain
    for ref_i, pose_i in _all_anchor_combinations(
        reference_chains, reference_entity_ids, pose_chains, pose_entity_ids
    ):
        # Superimpose the entire system
        # based on the anchor and chosen reference chain
        transform = _get_superimposition_transform(
            reference_chains[ref_i], pose_chains[pose_i]
        )
        superimposed_pose_centroids = transform.apply(pose_centroids)
        reference_indices, pose_indices = find_matching_centroids(
            reference_centroids,
            superimposed_pose_centroids,
            reference_entity_ids,
            pose_entity_ids,
        )
        rmsd = struc.rmsd(
            reference_centroids[reference_indices],
            superimposed_pose_centroids[pose_indices],
        )
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_transform = transform
            best_reference_indices = reference_indices
            best_pose_indices = pose_indices

    pose_chains = [best_transform.apply(chain) for chain in pose_chains]  # type: ignore[union-attr]
    return _match_using_chain_order(
        reference_chains,
        pose_chains,
        best_reference_indices,
        best_pose_indices,
        # Superimposition is already defined by at least two centroids,
        # as they leave no degrees of freedom
        superimpose=False if len(reference_centroids) > 1 else True,
    )


def _find_optimal_match_precise(
    reference_chains: list[struc.AtomArray],
    pose_chains: list[struc.AtomArray],
    min_sequence_identity: float,
    max_matches: int | None,
    allow_unmatched_entities: bool,
    use_entity_annotation: bool,
    use_structure_match: bool,
) -> tuple[struc.AtomArray, struc.AtomArray]:
    """
    Find matching atoms that minimize that minimize the all-atom RMSD between the pose
    and the reference.

    Parameters
    ----------
    reference_chains, pose_chains : list of AtomArray
        The reference and pose structure, separated into chains.
    min_sequence_identity : float
        The minimum sequence identity between two chains to be considered the same
        entity.
    max_matches : int, optional
        The maximum number of mappings to try.
    allow_unmatched_entities : bool, optional
        If set to ``True``, allow entire entities to be unmatched.
        This is useful if a pose is compared to a reference which may contain different
        molecules.
    use_entity_annotation : bool, optional
        If set to ``True``, use the ``entity_id`` annotation to determine which chains
        are the same entity and therefore are mappable to each other.
    use_structure_match : bool, optional
        If set to ``True``, use structure matching, i.e. isomorphism on the bond graph,
        to determine which small molecules are the same entity.
        Otherwise, match small molecules by residue name.
        Note that the structure match requires more computation time.

    Returns
    -------
    matched_reference, matched_pose : AtomArray
        The input atoms, where the chains and atoms within each chain are brought into
        the corresponding order.
    """
    if max_matches is not None and max_matches < 1:
        raise ValueError("Maximum number of mappings must be at least 1")

    best_rmsd = np.inf
    best_mapping: tuple[struc.AtomArray, struc.AtomArray] | None = None
    for it, (mapped_reference, mapped_pose) in enumerate(
        _all_global_mappings(
            reference_chains,
            pose_chains,
            min_sequence_identity,
            allow_unmatched_entities,
            use_entity_annotation,
            use_structure_match,
        )
    ):
        if max_matches is not None and it >= max_matches:
            break
        matched_ref_coord = mapped_reference.coord[mapped_reference.matched]
        matched_pose_coord = mapped_pose.coord[mapped_pose.matched]
        matched_pose_coord, _ = struc.superimpose(matched_ref_coord, matched_pose_coord)
        rmsd = struc.rmsd(matched_ref_coord, matched_pose_coord)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_mapping = (mapped_reference, mapped_pose)

    if best_mapping is None:
        raise UnmappableEntityError(
            "No chain in the pose can be mapped to a reference chain"
        )
    return best_mapping


def _all_global_mappings(
    reference_chains: list[struc.AtomArray],
    pose_chains: list[struc.AtomArray],
    min_sequence_identity: float = 0.95,
    allow_unmatched_entities: bool = False,
    use_entity_annotation: bool = False,
    use_structure_match: bool = False,
) -> Iterator[tuple[struc.AtomArray, struc.AtomArray]]:
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
    allow_unmatched_entities : bool, optional
        If set to ``True``, allow entire entities to be unmatched.
        This is useful if a pose is compared to a reference which may contain different
        molecules.
    use_entity_annotation : bool, optional
        If set to ``True``, use the ``entity_id`` annotation to determine which chains
        are the same entity and therefore are mappable to each other.
    use_structure_match : bool, optional
        If set to ``True``, use structure matching, i.e. isomorphism on the bond graph,
        to determine which small molecules are the same entity.
        Otherwise, match small molecules by residue name.
        Note that the structure match requires more computation time.

    Yields
    ------
    matched_reference, matched_pose : list of AtomArray
        The input chains, where the chains and atoms within each chain are brought into
        the corresponding order.

    Notes
    -----
    This functions tries all chain mappings of chain that are the same entity and
    within each small molecules tries all proper molecule permutations.
    """
    # Add the `matched` annotation in-place without copying
    # This is OK, as the chain `AtomArray`s were created in private
    # functions, so no user-provided data is modified
    for chain in itertools.chain(reference_chains, pose_chains):
        # Every atom is initialized as unmatched
        chain.set_annotation("matched", np.full(chain.array_length(), False))

    reference_entity_ids, pose_entity_ids = _assign_entity_ids(
        reference_chains,
        pose_chains,
        min_sequence_identity,
        allow_unmatched_entities,
        use_entity_annotation,
        use_structure_match,
    )
    for ref_chain_indices, pose_chain_indices in _all_chain_mappings(
        reference_entity_ids, pose_entity_ids
    ):
        # All possible intra-chain atom mappings for the current chain mapping
        # In case of symmetric small molecules, multiple mappings are possible
        # The outer list iterates over the matched chains
        # The inner list iterates over possible atom mappings within each chain
        all_intra_chain_mappings: list[
            list[tuple[struc.AtomArray, struc.AtomArray]]
        ] = []
        for ref_chain_i, pose_chain_i in zip(ref_chain_indices, pose_chain_indices):
            if (
                MoleculeType.of(reference_chains[ref_chain_i])
                == MoleculeType.SMALL_MOLECULE
            ):
                pose_mappings = _molecule_mappings(
                    reference_chains[ref_chain_i], pose_chains[pose_chain_i]
                )
                all_mapping_possibilities = []
                for pose_mapping in pose_mappings:
                    mapped_ref_chain = reference_chains[ref_chain_i].copy()
                    mapped_pose_chain = pose_chains[pose_chain_i][pose_mapping]
                    mapped_ref_chain.matched[:] = True
                    mapped_pose_chain.matched[:] = True
                    all_mapping_possibilities.append(
                        (mapped_ref_chain, mapped_pose_chain)
                    )
                all_intra_chain_mappings.append(all_mapping_possibilities)
            else:
                # For polymers there is only one mapping
                all_intra_chain_mappings.append(
                    [
                        _match_common_residues(
                            # Copy, as the `_match_common_residues()` modifies the
                            # `matched` annotation, so subsequent iteration would
                            # use a modified `matched` annotation
                            reference_chains[ref_chain_i].copy(),
                            pose_chains[pose_chain_i].copy(),
                        )
                    ]
                )

        unmapped_ref_chains = [
            reference_chains[i]
            for i in range(len(reference_chains))
            if i not in ref_chain_indices
        ]
        unmapped_pose_chains = [
            pose_chains[i]
            for i in range(len(pose_chains))
            if i not in pose_chain_indices
        ]
        # Create Cartesian product of intra-chain mappings over all chains
        for mapped_chains in itertools.product(*all_intra_chain_mappings):
            mapped_ref_chains = [chain for chain, _ in mapped_chains]
            mapped_pose_chains = [chain for _, chain in mapped_chains]
            yield (
                struc.concatenate(mapped_ref_chains + unmapped_ref_chains),
                struc.concatenate(mapped_pose_chains + unmapped_pose_chains),
            )


def _all_chain_mappings(
    reference_entity_ids: NDArray[np.int_],
    pose_entity_ids: NDArray[np.int_],
) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """
    Iterate over all possible mappings of chains between the reference and the pose
    that are the same entity.

    This function tries to map as many chains as possible, but if either structure
    contains more chains than the other for the same entity, chains may be unmapped
    (i.e. missing) in the returned mappings.

    Parameters
    ----------
    reference_entity_ids, pose_entity_ids : np.ndarray, shape=(n,), dtype=int
        The entity IDs of the chains.

    Yields
    ------
    reference_chain_indices, pose_chain_indices : np.ndarray, shape=(n,), dtype=int
        The indices of the chains in the reference and pose that are mapped to each
        other.
    """
    unique_entity_ids = np.unique(
        np.concatenate([reference_entity_ids, pose_entity_ids])
    )
    unique_entity_ids = unique_entity_ids[unique_entity_ids != _UNMAPPABLE_ENTITY_ID]
    if len(unique_entity_ids) == 0:
        raise UnmappableEntityError(
            "No chain in the pose can be mapped to a reference chain"
        )
    mappings_for_all_entities = []
    for entity_id in unique_entity_ids:
        # Find all chains that belong to the same entity
        ref_chain_indices = np.where(reference_entity_ids == entity_id)[0]
        pose_chain_indices = np.where(pose_entity_ids == entity_id)[0]
        # Pad the shorter index array
        max_length = max(len(ref_chain_indices), len(pose_chain_indices))
        padded_ref_chain_indices = np.full(max_length, _PADDING, dtype=int)
        padded_pose_chain_indices = np.full(max_length, _PADDING, dtype=int)
        padded_ref_chain_indices[: len(ref_chain_indices)] = ref_chain_indices
        padded_pose_chain_indices[: len(pose_chain_indices)] = pose_chain_indices
        ref_chain_indices = padded_ref_chain_indices
        pose_chain_indices = padded_pose_chain_indices
        # Create all possible mappings for chains of this entity
        # Mapping to the padding value means that the corresponding chain is not mapped
        possible_mappings_within_entity = []
        for permutated_pose_chain_indices in itertools.permutations(pose_chain_indices):
            possible_mappings_within_entity.append(
                np.stack([ref_chain_indices, permutated_pose_chain_indices], axis=1)
            )
        mappings_for_all_entities.append(possible_mappings_within_entity)

    # Create Cartesian product of all mappings
    for mapping_for_all_entities in itertools.product(*mappings_for_all_entities):
        mapping = np.concatenate(mapping_for_all_entities, axis=0)
        mapping = mapping[np.all(mapping != _PADDING, axis=1)]
        yield tuple(mapping.T)


def _match_using_chain_order(
    reference_chains: list[struc.AtomArray],
    pose_chains: list[struc.AtomArray],
    reference_chain_order: NDArray[np.int_],
    pose_chain_order: NDArray[np.int_],
    superimpose: bool = False,
) -> tuple[struc.AtomArray, struc.AtomArray]:
    """
    Given the order of corresponding chains between the reference and the pose,
    find the corresponding atom order within each chain and create structures with
    corresponding chain order and the ``matched`` annotation array.

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
    matched_reference, matched_pose: AtomArray
        The reference and pose, respectively, with the chains and atoms in the
        corresponding order.
        The structures have a ``matched`` annotation array, that is ``True`` for
        each atom that has a counterpart in the other structure.
    """
    # Add the `matched` annotation in-place without copying
    # This is OK, as the chain `AtomArray`s were created in private
    # functions, so no user-provided data is modified
    for chain in itertools.chain(reference_chains, pose_chains):
        # Every atom is initialized as unmatched
        chain.set_annotation("matched", np.full(chain.array_length(), False))

    handled_reference_chain_mask = np.full(len(reference_chains), False, dtype=bool)
    handled_pose_chain_mask = np.full(len(pose_chains), False, dtype=bool)
    matched_reference_chains = []
    matched_pose_chains = []
    for ref_i, pose_i in zip(reference_chain_order, pose_chain_order):
        if MoleculeType.of(reference_chains[ref_i]) == MoleculeType.SMALL_MOLECULE:
            ref_atom_order, pose_atom_order = _find_optimal_molecule_permutation(
                reference_chains[ref_i],
                pose_chains[pose_i],
                superimpose=superimpose,
            )
            ref_chain = reference_chains[ref_i][ref_atom_order]
            pose_chain = pose_chains[pose_i][pose_atom_order]
            ref_chain.matched[:] = True
            pose_chain.matched[:] = True
            matched_reference_chains.append(ref_chain)
            matched_pose_chains.append(pose_chain)
        else:
            ref_chain, pose_chain = _match_common_residues(
                reference_chains[ref_i], pose_chains[pose_i]
            )
            matched_reference_chains.append(ref_chain)
            matched_pose_chains.append(pose_chain)
        handled_reference_chain_mask[ref_i] = True
        handled_pose_chain_mask[pose_i] = True

    return (
        struc.concatenate(
            matched_reference_chains
            # The unmatched chains are appended to the end
            + [reference_chains[i] for i in np.where(~handled_reference_chain_mask)[0]]
        ),
        struc.concatenate(
            matched_pose_chains
            # The unmatched chains are appended to the end
            + [pose_chains[i] for i in np.where(~handled_pose_chain_mask)[0]]
        ),
    )


def _match_common_residues(
    reference: struc.AtomArray, pose: struc.AtomArray
) -> tuple[struc.AtomArray, struc.AtomArray]:
    """
    Find common residues (and the common atoms) in two polymer chains.

    Parameters
    ----------
    reference, pose : struc.AtomArray, shape=(n,)
        The reference and pose polymer chain whose common atoms are to be found.

    Returns
    -------
    matched_reference, matched_pose : struc.AtomArray, shape=(n,)
        Reordered versions of the input structures, where the common atoms are in the
        corresponding order and unique atoms in each structure get annotated with
        ``matched=False``.
    """
    # Shortcut if the structures already match perfectly atom-wise
    if _is_matched(reference, pose, _ANNOTATIONS_FOR_ATOM_MATCHING):
        reference.matched[:] = True
        pose.matched[:] = True
        return reference, pose

    reference_sequence = struc.to_sequence(reference)[0][0]
    pose_sequence = struc.to_sequence(pose)[0][0]
    alignment = align.align_optimal(
        reference_sequence,
        pose_sequence,
        _IDENTITY_MATRIX[MoleculeType.of(reference)],
        # We get mismatches due to cropping, not due to evolution
        # -> linear gap penalty makes most sense
        gap_penalty=-1,
        terminal_penalty=False,
        max_number=1,
    )[0]
    # Remove gaps -> crop structures to common residues
    alignment.trace = alignment.trace[(alignment.trace != -1).all(axis=1)]

    # Atom masks that are True for atoms in residues that are common in both structures
    ref_aligned_mask = _get_mask_from_alignment_trace(reference, alignment.trace[:, 0])
    pose_aligned_mask = _get_mask_from_alignment_trace(pose, alignment.trace[:, 1])

    # Within the atoms of aligned residues, select only common atoms
    ref_order, pose_order, ref_common_mask, pose_common_mask = _find_atom_intersection(
        reference[ref_aligned_mask], pose[pose_aligned_mask]
    )
    global_ref_order = np.arange(reference.array_length())
    global_pose_order = np.arange(pose.array_length())
    global_ref_order[ref_aligned_mask] = global_ref_order[ref_aligned_mask][ref_order]
    global_pose_order[pose_aligned_mask] = global_pose_order[pose_aligned_mask][
        pose_order
    ]
    reference = reference[global_ref_order]
    pose = pose[global_pose_order]
    reference.matched[ref_aligned_mask] = ref_common_mask
    pose.matched[pose_aligned_mask] = pose_common_mask

    return reference, pose


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
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Find the intersection of two structures, i.e. the set of equivalent atoms.

    Parameters
    ----------
    reference, pose : AtomArray
        The reference and pose chain, respectively.

    Returns
    -------
    ref_order, pose_order : ndarray, shape=(n,), dtype=int
        The reference and pose indices that bring the input structures into a common
        atom order.
    intersection_ref_mask, intersection_pose_mask : ndarray, shape=(n,), dtype=bool
        The mask that is `True` for each atom in the sorted atom order, that appears
        in both structures.

    Notes
    -----
    The order is not necessarily canonical, as the atoms are simply sorted based on the
    given annotations.
    The important requirement is that the order is the same for both structures.
    """
    # Shortcut if the structures already match perfectly atom-wise
    if _is_matched(reference, pose, _ANNOTATIONS_FOR_ATOM_MATCHING):
        return (
            np.arange(reference.array_length()),
            np.arange(reference.array_length()),
            np.full(reference.array_length(), True),
            np.full(reference.array_length(), True),
        )

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
    # Atom ordering might not be same -> sort
    ref_order = np.argsort(ref_annotations)
    pose_order = np.argsort(pose_annotations)
    # Identify the intersection of the two annotation arrays
    intersection_ref_mask = np.isin(ref_annotations[ref_order], pose_annotations)
    intersection_pose_mask = np.isin(pose_annotations[pose_order], ref_annotations)

    return ref_order, pose_order, intersection_ref_mask, intersection_pose_mask


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
        # If still no match is found, conformation is not the problem
        # -> Check if bond graph is the problem
        if np.array_equal(reference.element, pose.element):
            # If the elements are the same, probably some simple
            # incompatible bonds are the problem
            # -> Assume that the atom order is the same in reference and pose
            mappings = [np.arange(pose.array_length())]
            warnings.warn(
                "Incompatible bond graph between pose and reference small molecule",
                GraphMatchWarning,
            )
        else:
            # They are probably different incompatible molecules
            # This is a user error, as different molecules should have different
            # residue names
            raise UnmappableEntityError(
                "No atom mapping found between pose and reference small molecule "
                f"'{reference.res_name[0]}'"
            )

    # Convert tuples to proper index arrays
    return [np.asarray(mapping) for mapping in mappings]


def _assign_entity_ids(
    reference_chains: list[struc.AtomArray],
    pose_chains: list[struc.AtomArray],
    min_sequence_identity: float,
    allow_unmatched_entities: bool = False,
    use_entity_annotation: bool = False,
    use_structure_match: bool = False,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Assign a unique entity ID to each distinct chain.

    This means that two chains with the same entity ID have sufficient sequence
    identity or in case of small molecules have the same ``res_name`` (or the same
    bond graph if `use_structure_match` is set to ``True``).

    Parameters
    ----------
    reference_chains, pose_chains : list of struc.AtomArray, length=p or length=q
        The reference and pose chains, respectively.
    min_sequence_identity : float
        The minimum sequence identity between two chains to be considered the same
        entity.
    allow_unmatched_entities : bool, optional
        If set to ``True``, allow the reference and pose to have different entities.
        Otherwise an :class:`UnmappableEntityError` is raised.
    use_entity_annotation : bool, optional
        If set to ``True``, use the ``entity_id`` annotation to determine which chains
        are the same entity and therefore are mappable to each other.
    use_structure_match : bool, optional
        If set to ``True``, use structure matching, i.e. isomorphism on the bond graph,
        to determine which small molecules are the same entity.
        Otherwise, match small molecules by residue name.
        Note that the structure match requires more computation time.

    Returns
    -------
    reference_entity_ids, pose_entity_ids : np.ndarray, shape=(p,) or shape=(q,), dtype=int
        The entity IDs.
        Chains that are not mappable between reference and pose get the ID ``-1``.
    """
    # Assign reference and pose entity IDs in a single call
    # in order to assign the same ID to corresponding chains between reference and pose
    entity_ids = _assign_entity_ids_to_chains(
        reference_chains + pose_chains,
        min_sequence_identity,
        use_entity_annotation,
        use_structure_match,
    )
    max_entity_id = np.max(entity_ids)

    # Split the entity IDs again
    reference_entity_ids = entity_ids[: len(reference_chains)]
    pose_entity_ids = entity_ids[len(reference_chains) :]
    # In the worst case, the number of distinct entity IDs
    # is equal to the number of chains -> use this as 'minlength'
    reference_entity_id_counts = np.bincount(
        reference_entity_ids, minlength=max_entity_id + 1
    )
    pose_entity_id_counts = np.bincount(pose_entity_ids, minlength=max_entity_id + 1)
    if (
        not allow_unmatched_entities
        and (reference_entity_id_counts != pose_entity_id_counts).any()
    ):
        raise UnmappableEntityError(
            "Reference and pose have different entities or a different multiplicity"
        )
    # Assign an unmappable entity ID to IDs that only appear in one structure
    reference_entity_ids[pose_entity_id_counts[reference_entity_ids] == 0] = (
        _UNMAPPABLE_ENTITY_ID
    )
    pose_entity_ids[reference_entity_id_counts[pose_entity_ids] == 0] = (
        _UNMAPPABLE_ENTITY_ID
    )

    return reference_entity_ids, pose_entity_ids


def _assign_entity_ids_to_chains(
    chains: list[struc.AtomArray],
    min_sequence_identity: float,
    use_entity_annotation: bool,
    use_structure_match: bool,
) -> NDArray[np.int_]:
    """
    Assign a unique entity ID to each distinct chain.

    This means that two chains with the same entity ID have sufficient sequence
    identity or in case of small molecules have the same ``res_name`` (or the same
    bond graph if `use_structure_match` is set to ``True``).

    Parameters
    ----------
    chains : list of struc.AtomArray, length=n
        The chains to assign entity IDs to.
    min_sequence_identity : float
        The minimum sequence identity between two chains to be considered the same
        entity.
    use_entity_annotation : bool, optional
        If set to ``True``, use the ``entity_id`` annotation to determine which chains
        are the same entity and therefore are mappable to each other.
    use_structure_match : bool, optional
        If set to ``True``, use structure matching, i.e. isomorphism on the bond graph,
        to determine which small molecules are the same entity.
        Otherwise, match small molecules by residue name.
        Note that the structure match requires more computation time.


    Returns
    -------
    entity_ids : np.ndarray, shape=(n,), dtype=int
        The entity IDs.
    """
    entity_ids: list[int] = []
    if use_entity_annotation:
        for chain in chains:
            if "entity_id" not in chain.get_annotation_categories():
                raise struc.BadStructureError(
                    "'use_entity_annotation=True' requires the 'entity_id' annotation"
                )
            entity_id = chain.entity_id[0].item()
            if not np.all(chain.entity_id == entity_id):
                raise struc.BadStructureError(
                    "'entity_id' annotation must be the same for all atoms in a chain"
                )
            entity_ids.append(chain.entity_id[0].item())
        return np.array(entity_ids, dtype=int)

    molecule_types = [MoleculeType.of(chain) for chain in chains]
    sequences = [
        struc.to_sequence(chain)[0][0]
        if molecule_type != MoleculeType.SMALL_MOLECULE
        else None
        for chain, molecule_type in zip(chains, molecule_types)
    ]
    if use_structure_match:
        molecules = [
            _to_mol(chain) if molecule_type == MoleculeType.SMALL_MOLECULE else None
            for chain, molecule_type in zip(chains, molecule_types)
        ]

    current_entity_id = 0
    for i, (chain, sequence, molecule_type) in enumerate(
        zip(chains, sequences, molecule_types)
    ):
        for j in range(i):
            if molecule_type != molecule_types[j]:
                # Cannot match different molecule types to each other
                continue
            elif molecule_type == MoleculeType.SMALL_MOLECULE:
                # Small molecule case
                if use_structure_match:
                    # It is only a complete structure match,
                    # if i is a non-strict subset of j and j is a non-strict subset of i
                    if (
                        molecules[i].HasSubstructMatch(molecules[j]) and  # type: ignore[union-attr]
                        molecules[j].HasSubstructMatch(molecules[i])  # type: ignore[union-attr]
                    ):  # fmt: skip
                        entity_ids.append(entity_ids[j])
                        break
                else:
                    # Match small molecules by residue name
                    if chain.res_name[0] == chains[j].res_name[
                        0
                    ] and _equal_composition([chain, chains[j]]):
                        entity_ids.append(entity_ids[j])
                        break
            else:
                # Match polymer chains by sequence identity
                alignment = align.align_optimal(
                    sequence,
                    sequences[j],
                    _IDENTITY_MATRIX[molecule_type],
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

    return np.array(entity_ids, dtype=int)


def _all_anchor_combinations(
    reference_chains: list[struc.AtomArray],
    reference_entity_ids: NDArray[np.int_],
    pose_chains: list[struc.AtomArray],
    pose_entity_ids: NDArray[np.int_],
) -> Iterator[tuple[int, int]]:
    """
    Choose the reference and pose anchor chain for the heuristic chain matching.

    The most preferable chain is the one with the least multiplicity and the longest
    sequence.

    Parameters
    ----------
    chains : list of struc.AtomArray, length=n
        The chains to choose from.
    entity_ids : ndarray, shape=(n,), dtype=int
        The entity IDs of the chains.
        Used to determine the multiplicity of each chain.

    Yields
    ------
    reference_i, pose_i : int
        The putative anchor indices.
    """
    mappable_mask = pose_entity_ids != _UNMAPPABLE_ENTITY_ID
    if not mappable_mask.any():
        raise UnmappableEntityError(
            "No chain in the pose can be mapped to a reference chain"
        )
    polymer_chain_mask = np.array(
        [
            not MoleculeType.of(chain) != MoleculeType.SMALL_MOLECULE
            for chain in pose_chains
        ]
    )
    valid_anchor_mask = polymer_chain_mask & mappable_mask
    if not valid_anchor_mask.any():
        # No mappable polymer chains
        # -> Simply use the first mappable small molecule as anchor
        anchor_entity_id = pose_entity_ids[np.where(mappable_mask)[0][0]]
    else:
        valid_anchor_indices = np.where(valid_anchor_mask)[0]
        polymer_entity_ids = pose_entity_ids[valid_anchor_indices]
        multiplicities_of_entity_ids = np.bincount(polymer_entity_ids)
        multiplicities = multiplicities_of_entity_ids[polymer_entity_ids]
        least_multiplicity_indices = np.where(multiplicities == np.min(multiplicities))[
            0
        ]
        # Use the sequence length as tiebreaker
        sequence_lengths = np.array([len(pose_chains[i]) for i in valid_anchor_indices])
        # Only consider the lengths of the preselected chains
        largest_length = np.max(sequence_lengths[least_multiplicity_indices])
        largest_length_indices = np.where(sequence_lengths == largest_length)[0]
        best_pose_anchors = np.intersect1d(
            least_multiplicity_indices, largest_length_indices
        )
        anchor_entity_id = pose_entity_ids[valid_anchor_mask][best_pose_anchors[0]]

    # Check if there is a 1-to-1 correspondence of the anchor entity between
    # the reference and pose
    if np.count_nonzero(reference_entity_ids == anchor_entity_id) == np.count_nonzero(
        pose_entity_ids == anchor_entity_id
    ):
        # It is sufficient to keep the pose anchor fixed and iterate over the reference
        pose_i = np.where(pose_entity_ids == anchor_entity_id)[0][0]
        for ref_i in range(len(reference_chains)):
            if reference_entity_ids[ref_i] == anchor_entity_id:
                yield ref_i, pose_i
    else:
        # Otherwise an anchor chain in the pose might be missing in the reference
        # -> We need to iterate over all pose anchors as well
        for pose_i in np.where(pose_entity_ids == anchor_entity_id)[0]:
            for ref_i in range(len(reference_chains)):
                if reference_entity_ids[ref_i] == anchor_entity_id:
                    yield ref_i, pose_i


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
    if MoleculeType.of(reference_chain) == MoleculeType.SMALL_MOLECULE:
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
            _IDENTITY_MATRIX[MoleculeType.of(reference_chain)],
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


def _equal_composition(molecules: list[struc.AtomArray]) -> bool:
    """
    Check if the element composition is the same in both molecules.

    The atoms in each molecule may be in a different order.

    Parameters
    ----------
    molecules : list[AtomArray]
        The atoms to compare.

    Returns
    -------
    equal : bool
        True, if the element composition is the same in both molecules.
    """
    element_counters: list[Counter[str]] = [Counter() for _ in molecules]
    for i, molecule in enumerate(molecules):
        for element in molecule.element:
            element_counters[i][element] += 1
    return all(element_counters[0] == counter for counter in element_counters[1:])


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
