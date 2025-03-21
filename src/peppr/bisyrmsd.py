__all__ = [
    "bisy_rmsd",
]

import itertools
import biotite.structure as struc
import numpy as np
from numpy.typing import NDArray
from peppr.common import is_small_molecule
from peppr.dockq import get_contact_residues

_REPRESENTATIVE_ATOMS = ["CA", "C3'"]
_BACKBONE_ATOMS = (
    "CA",
    "C",
    "N",
    "P",
    "OP1",
    "OP2",
    "O2'",
    "O3'",
    "O4'",
    "O5'",
    "C1'",
    "C2'",
    "C3'",
    "C4'",
    "C5'",
)


def bisy_rmsd(
    reference: struc.AtomArray,
    pose: struc.AtomArray,
    inclusion_radius: float = 4.0,
    outlier_distance: float = 3.0,
    max_iterations: int = 5,
    min_anchors: int = 3,
) -> float:
    """
    Compute the *Binding-Site Superposed, Symmetry-Corrected Pose RMSD* (BiSyRMSD) for
    the given PLI complex as defined in [1]_.

    Parameters
    ----------
    reference, pose : AtomArray
        The reference and pose of the PLI complex.
    inclusion_radius : float, optional
        All residues where at least one heavy atom is within this radius of a heavy
        ligand atom, are considered part of the binding site.
        The default value is taken from [1]_.
    outlier_distance : float, optional
        The binding sites of the reference and pose are superimposed iteratively.
        In each iteration, atoms with a distance of more than this value are considered
        outliers and are removed in the next iteration.
        The default value is taken from [1]_.
        To disable outlier removal, set this value to ``inf``.
    max_iterations : int, optional
        The maximum number of iterations for the superimposition.
        The default value is taken from [1]_.
    min_anchors : int, optional
        The minimum number of anchors to use for the superimposition.
        If less than this number of anchors are present, the superimposition is
        performed on all interface backbone atoms.
        The default value is taken from [1]_.

    Returns
    -------
    float
        The *BiSyRMSD*.

    References
    ----------
    .. [1] https://doi.org/10.1002/prot.26601
    """
    receptor_mask = ~reference.hetero
    if not any(receptor_mask):
        # No receptor present
        return np.nan
    reference_ligand_chains = []
    pose_ligand_chains = []
    for start_i, stop_i in itertools.pairwise(
        struc.get_chain_starts(reference, add_exclusive_stop=True)
    ):
        reference_chain = reference[start_i:stop_i]
        if is_small_molecule(reference_chain):
            reference_ligand_chains.append(reference_chain)
            pose_ligand_chains.append(pose[start_i:stop_i])
    if len(reference_ligand_chains) == 0:
        # No ligand present
        return np.nan

    reference_ligand_coord = [chain.coord for chain in reference_ligand_chains]
    superimposed_pose_ligand_coord = [
        _superimpose_binding_site(
            reference[receptor_mask],
            pose[receptor_mask],
            reference_ligand_chain,
            inclusion_radius,
            outlier_distance,
            max_iterations,
            min_anchors,
        ).apply(pose_ligand_chain.coord)
        for reference_ligand_chain, pose_ligand_chain in zip(
            reference_ligand_chains, pose_ligand_chains
        )
    ]

    # Compute the RMSD for each ligand and take the mean
    return np.mean(
        [
            struc.rmsd(ref_coord, mod_coord).item()
            for ref_coord, mod_coord in zip(
                reference_ligand_coord, superimposed_pose_ligand_coord, strict=True
            )
        ]
    ).item()


def _superimpose_binding_site(
    reference_receptor: struc.AtomArray,
    pose_receptor: struc.AtomArray,
    reference_ligand: struc.AtomArray,
    inclusion_radius: float,
    outlier_distance: float,
    max_iterations: int,
    min_anchors: int,
) -> struc.AffineTransformation:
    """
    Get a transformation that superimposes the binding site of the pose receptor onto
    the reference receptor.

    Parameters
    ----------
    reference_receptor, pose_receptor : AtomArray
        The reference and pose receptor.
    reference_ligand : AtomArray
        The reference ligand.
        The binding site is determined by the residues in contact with the ligand atoms.
    inclusion_radius : float, optional
        All residues where at least one heavy atom is within this radius of a heavy
        ligand atom, are considered part of the binding site.
    outlier_distance : float, optional
        The binding sites of the reference and pose are superimposed iteratively.
        In each iteration, atoms with a distance of more than this value are considered
        outliers and are removed in the next iteration.
        To disable outlier removal, set this value to ``inf``.
    max_iterations : int, optional
        The maximum number of iterations for the superimposition.
    min_anchors : int, optional
        The minimum number of anchors to use for the superimposition.
        If less than this number of anchors are present, the superimposition is
        performed on all interface backbone atoms.

    Returns
    -------
    AffineTransformation
        The transformation that superimposes the binding site of the pose receptor onto
        the reference receptor.
    """
    receptor_contacts = np.unique(
        get_contact_residues(
            reference_receptor, reference_ligand, cutoff=inclusion_radius
        )[:, 0]
    )
    if len(receptor_contacts) >= min_anchors:
        interface_mask = (
            struc.get_residue_masks(reference_receptor, receptor_contacts).any(axis=0)
            # Use one atom per residue as superimposition anchor
            & np.isin(reference_receptor.atom_name, _REPRESENTATIVE_ATOMS)
        )
        transform = _superimpose_without_outliers(
            reference_receptor.coord[interface_mask],
            pose_receptor.coord[interface_mask],
            outlier_distance,
            max_iterations,
            min_anchors,
        )
    else:
        # Use all interface backbone atoms as superimposition anchor
        interface_mask = np.isin(
            reference_receptor.res_id, receptor_contacts
        ) & np.isin(reference_receptor.atom_name, _BACKBONE_ATOMS)
        _, transform = struc.superimpose(
            reference_receptor.coord[interface_mask],
            pose_receptor.coord[interface_mask],
        )
    return transform


def _superimpose_without_outliers(
    reference_coord: NDArray[np.floating],
    pose_coord: NDArray[np.floating],
    outlier_distance: float,
    max_iterations: int,
    min_anchors: int,
) -> struc.AffineTransformation:
    """
    Get a transformation that superimposes the given pose coordinates onto the
    reference coordinates.

    Outliers are iteratively removed until no outliers are left.

    Parameters
    ----------
    reference_coord, pose_coord : ndarray, shape=(n,3)
        The reference and pose coordinates.
    outlier_distance : float
        In each iteration, atoms with a distance of more than this value are considered
        outliers and are removed in the next iteration.
    max_iterations : int
        The maximum number of iterations.
    min_anchors : int
        The minimum number of anchors to use for the superimposition.
        If less than this number of anchors remain after outlier removal, the
        superimposition is performed on all interface backbone atoms.

    Returns
    -------
    biotite.structure.AffineTransformation
        The transformation that superimposes the pose onto the reference.
    """
    if max_iterations < 1:
        raise ValueError("Maximum number of iterations must be at least 1")

    # Before iterative refinement, all anchors are included
    # 'inlier' is the opposite of 'outlier'
    updated_inlier_mask = np.ones(reference_coord.shape[-2], dtype=bool)

    for _ in range(max_iterations):
        # Run superimposition
        inlier_mask = updated_inlier_mask
        filtered_reference_coord = reference_coord[..., inlier_mask, :]
        filtered_pose_coord = pose_coord[..., inlier_mask, :]
        superimposed_coord, transform = struc.superimpose(
            filtered_reference_coord, filtered_pose_coord
        )

        # Find outliers
        distance = struc.distance(filtered_reference_coord, superimposed_coord)
        updated_inlier_mask = inlier_mask.copy()
        # Distance was only calculated for the existing inliers
        # -> update the mask only for these atoms
        updated_inlier_mask[updated_inlier_mask] = distance <= outlier_distance

        if np.all(updated_inlier_mask):
            # No outliers anymore -> early termination
            break
        if np.count_nonzero(updated_inlier_mask) < min_anchors:
            # Less than min_anchors anchors would be left
            # -> revert to superimposition of all coordinates
            _, transform = struc.superimpose(reference_coord, pose_coord)
            break

    return transform
