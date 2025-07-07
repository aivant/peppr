"""
Calculate DockQ for a single pair of receptor and ligand.
"""

__all__ = [
    "get_contact_residues",
    "dockq",
    "pocket_aligned_lrmsd",
    "lrmsd",
    "irmsd",
    "fnat",
    "DockQ",
    "NoContactError",
]

from dataclasses import dataclass, field
from typing import overload
import biotite.structure as struc
import numpy as np
from numpy.typing import NDArray
from peppr.common import is_small_molecule

BACKBONE_ATOMS = (
    "CA",
    "C",
    "N",
    "O",
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


class NoContactError(Exception):
    pass


@dataclass(frozen=True)
class DockQ:
    """
    Result of a *DockQ* calculation.

    If multiple poses were used to calculate *DockQ*, the attributes are arrays.

    Attributes
    ----------
    fnat : float or ndarray, dtype=float
        The fraction of reference contacts found in the pose relative to the total
        number of reference contacts.
    fnonnat : float or ndarray, dtype=float
        The fraction of non-reference contacts found in the pose relative to the total
        number of pose contacts.
    irmsd : float or ndarray, dtype=float
        The interface RMSD.
    lrmsd : float or ndarray, dtype=float
        The ligand RMSD.
    score : float or ndarray, dtype=float
        The DockQ score.
    n_poses : int or None
        The number of poses for which the *DockQ* was calculated.
        `None`, if the *DockQ* was calculated for an `AtomArray`.
    pose_receptor_index, pose_ligand_index, reference_receptor_index, reference_ligand_index : int or None
        The indices of the pose and reference chain that were included for *DockQ*
        computation.
        Only set, if called from `global_dockq()`.
    """

    fnat: float | NDArray[np.floating]
    fnonnat: float | NDArray[np.floating]
    irmsd: float | NDArray[np.floating]
    lrmsd: float | NDArray[np.floating]
    score: float | NDArray[np.floating] = field(init=False)
    n_poses: int | None = field(init=False)
    pose_receptor_index: int | None = None
    pose_ligand_index: int | None = None
    reference_receptor_index: int | None = None
    reference_ligand_index: int | None = None

    def __post_init__(self) -> None:
        score = np.nanmean(
            [self.fnat, _scale(self.irmsd, 1.5), _scale(self.lrmsd, 8.5)], axis=0
        )
        n_poses = None if np.isscalar(score) else len(score)
        super().__setattr__("score", score)
        super().__setattr__("n_poses", n_poses)

    def for_pose(self, pose_index: int) -> "DockQ":
        """
        Get the DockQ results for a specific pose index.

        Parameters
        ----------
        pose_index : int
            The index of the pose for which the DockQ results should be retrieved.

        Returns
        -------
        DockQ
            The DockQ results for the specified pose index.

        Raises
        ------
        IndexError
            If the `GlobalDockQ` object was computed for a single pose,
            i.e. `n_poses` is `None`.
        """
        if self.n_poses is None:
            raise IndexError("DockQ was computed for a single pose")
        return DockQ(
            self.fnat[pose_index].item(),  # type: ignore[index]
            self.fnonnat[pose_index].item(),  # type: ignore[index]
            self.irmsd[pose_index].item(),  # type: ignore[index]
            self.lrmsd[pose_index].item(),  # type: ignore[index]
            pose_receptor_index=self.pose_receptor_index,
            pose_ligand_index=self.pose_ligand_index,
            reference_receptor_index=self.reference_receptor_index,
            reference_ligand_index=self.reference_ligand_index,
        )


def dockq(
    reference_receptor: struc.AtomArray,
    reference_ligand: struc.AtomArray,
    pose_receptor: struc.AtomArray | struc.AtomArrayStack,
    pose_ligand: struc.AtomArray | struc.AtomArrayStack,
    as_peptide: bool = False,
) -> DockQ:
    """
    Compute *DockQ* for a single pair of receptor and ligand in both, the pose and
    reference structure.

    Parameters
    ----------
    reference_receptor, reference_ligand : AtomArray
        The reference receptor and ligand.
    pose_receptor, pose_ligand : AtomArray or AtomArrayStack
        The pose receptor and ligand.
        Multiple poses can be provided.
    as_peptide : bool
        If set to true, the chains are treated as CAPRI peptides.

    Returns
    -------
    DockQ
        The DockQ result.
        If multiple poses are provided, the `DockQ` attributes are arrays.

    Notes
    -----
    If the ligand is a small molecule, an associated `BondList` is required in
    `pose_ligand` and `reference_ligand` for mapping the atoms between them.

    Examples
    --------

    Single chains as expected as input.

    >>> pose_receptor = pose_complex[pose_complex.chain_id == "C"]
    >>> pose_ligand = pose_complex[pose_complex.chain_id == "B"]
    >>> reference_receptor = reference_complex[reference_complex.chain_id == "C"]
    >>> reference_ligand = reference_complex[reference_complex.chain_id == "B"]
    >>> dockq_result = dockq(pose_receptor, pose_ligand, reference_receptor, reference_ligand)
    >>> print(f"{dockq_result.fnat:.2f}")
    0.50
    >>> print(f"{dockq_result.irmsd:.2f}")
    2.10
    >>> print(f"{dockq_result.lrmsd:.2f}")
    8.13
    >>> print(f"{dockq_result.score:.2f}")
    0.45
    """
    undefined = (
        np.nan
        if isinstance(pose_receptor, struc.AtomArray)
        else np.full(pose_receptor.stack_depth(), np.nan)
    )

    if as_peptide:
        if any(
            [
                is_small_molecule(chain)
                for chain in (
                    pose_receptor,
                    pose_ligand,
                    reference_receptor,
                    reference_ligand,
                )
            ]
        ):
            raise ValueError("'as_peptide' is true, but the chains are not peptides")

    if is_small_molecule(pose_ligand):
        # For small molecules DockQ is only based on the pocket-aligned ligand RMSD
        lrmsd_ = pocket_aligned_lrmsd(
            reference_receptor, reference_ligand, pose_receptor, pose_ligand
        )
        return DockQ(undefined, undefined, undefined, lrmsd_)

    else:
        fnat_, fnonnat_ = fnat(
            reference_receptor,
            reference_ligand,
            pose_receptor,
            pose_ligand,
            as_peptide,
        )
        if np.isnan(fnat_).any():
            # No contact between the chains -> DockQ is not defined
            return DockQ(undefined, undefined, undefined, undefined)

        irmsd_ = irmsd(
            reference_receptor,
            reference_ligand,
            pose_receptor,
            pose_ligand,
            as_peptide,
        )

        lrmsd_ = lrmsd(reference_receptor, reference_ligand, pose_receptor, pose_ligand)

        return DockQ(fnat_, fnonnat_, irmsd_, lrmsd_)


def pocket_aligned_lrmsd(
    reference_receptor: struc.AtomArray,
    reference_ligand: struc.AtomArray,
    pose_receptor: struc.AtomArray | struc.AtomArrayStack,
    pose_ligand: struc.AtomArray | struc.AtomArrayStack,
) -> float | NDArray[np.floating]:
    """
    Compute the pocket-aligned RMSD part of the DockQ score for small molecules.

    Parameters
    ----------
    reference_receptor, reference_ligand : AtomArray
        The reference receptor and ligand.
    pose_receptor, pose_ligand : AtomArray
        The pose receptor and ligand.

    Returns
    -------
    float or ndarray, dtype=float
        The pocket-aligned RMSD.
    """
    reference_contacts = get_contact_residues(
        reference_receptor,
        reference_ligand,
        cutoff=10.0,
    )
    if len(reference_contacts) == 0:
        # if there're no contacts between the two chains, no lrmsd is calculated
        return (
            np.full(shape=len(pose_ligand), fill_value=np.nan)
            if isinstance(pose_ligand, struc.AtomArrayStack)
            else np.nan
        )
    # Create mask which is True for all backbone atoms in contact receptor residues
    interface_mask = struc.get_residue_masks(
        reference_receptor, reference_contacts[:, 0]
    ).any(axis=0) & np.isin(reference_receptor.atom_name, BACKBONE_ATOMS)
    # Use interface backbone coordinates for pocket-aligned superimposition
    _, transform = struc.superimpose(
        reference_receptor.coord[interface_mask],
        pose_receptor.coord[..., interface_mask, :],
    )
    # Use the superimposed coordinates for RMSD calculation between ligand atoms
    lrmsd = struc.rmsd(reference_ligand.coord, transform.apply(pose_ligand.coord))
    return lrmsd.item() if np.isscalar(lrmsd) else lrmsd  # type: ignore[union-attr]


def lrmsd(
    reference_receptor: struc.AtomArray,
    reference_ligand: struc.AtomArray,
    pose_receptor: struc.AtomArray | struc.AtomArrayStack,
    pose_ligand: struc.AtomArray | struc.AtomArrayStack,
) -> float | NDArray[np.floating]:
    """
    Compute the ligand RMSD part of the DockQ score.

    Parameters
    ----------
    reference_receptor, reference_ligand : AtomArray
        The reference receptor and ligand.
    pose_receptor, pose_ligand : AtomArray
        The pose receptor and ligand.

    Returns
    -------
    float or ndarray, dtype=float
        The ligand RMSD.
    """
    receptor_relevant_mask = np.isin(pose_receptor.atom_name, BACKBONE_ATOMS)
    if is_small_molecule(pose_ligand):
        # For small molecules include all heavy atoms
        ligand_relevant_mask = np.full(pose_ligand.array_length(), True)
    else:
        ligand_relevant_mask = np.isin(pose_ligand.atom_name, BACKBONE_ATOMS)

    pose_receptor_coord = pose_receptor.coord[..., receptor_relevant_mask, :]
    pose_ligand_coord = pose_ligand.coord[..., ligand_relevant_mask, :]
    reference_receptor_coord = reference_receptor.coord[receptor_relevant_mask]
    reference_ligand_coord = reference_ligand.coord[ligand_relevant_mask]
    _, transform = struc.superimpose(
        reference_receptor_coord,
        pose_receptor_coord,
    )
    superimposed_ligand_coord = transform.apply(pose_ligand_coord)
    lrmsd = struc.rmsd(reference_ligand_coord, superimposed_ligand_coord)
    return lrmsd.item() if np.isscalar(lrmsd) else lrmsd  # type: ignore[union-attr]


def irmsd(
    reference_receptor: struc.AtomArray,
    reference_ligand: struc.AtomArray,
    pose_receptor: struc.AtomArray | struc.AtomArrayStack,
    pose_ligand: struc.AtomArray | struc.AtomArrayStack,
    as_peptide: bool = False,
) -> float | NDArray[np.floating]:
    """
    Compute the interface RMSD part of the DockQ score.

    Parameters
    ----------
    reference_receptor, reference_ligand : AtomArray
        The reference receptor and ligand.
    pose_receptor, pose_ligand : AtomArray or AtomArrayStack
        The pose receptor and ligand.
    as_peptide : bool
        If set to true, the chains are treated as CAPRI peptides.

    Returns
    -------
    float or ndarray, dtype=float
        The interface RMSD.
    """
    if as_peptide:
        cutoff = 8.0
        receptor_mask = _mask_either_or(reference_receptor, "CB", "CA")
        ligand_mask = _mask_either_or(reference_ligand, "CB", "CA")
    else:
        cutoff = 10.0
        receptor_mask = None
        ligand_mask = None
    reference_contacts = get_contact_residues(
        reference_receptor,
        reference_ligand,
        cutoff,
        receptor_mask,
        ligand_mask,
    )

    if len(reference_contacts) == 0:
        # if there're no contacts between the two chains,
        # no irmsd is calculated
        return (
            np.full(shape=len(pose_ligand), fill_value=np.nan)
            if isinstance(pose_ligand, struc.AtomArrayStack)
            else np.nan
        )

    # Create mask which is True for all backbone atoms in contact residues
    receptor_backbone_interface_mask = struc.get_residue_masks(
        reference_receptor, reference_contacts[:, 0]
    ).any(axis=0) & np.isin(reference_receptor.atom_name, BACKBONE_ATOMS)
    ligand_backbone_interface_mask = struc.get_residue_masks(
        reference_ligand, reference_contacts[:, 1]
    ).any(axis=0) & np.isin(reference_ligand.atom_name, BACKBONE_ATOMS)

    # Get the coordinates of interface backbone atoms
    pose_interface_coord = np.concatenate(
        [
            pose_receptor.coord[..., receptor_backbone_interface_mask, :],
            pose_ligand.coord[..., ligand_backbone_interface_mask, :],
        ],
        axis=-2,
    )
    reference_interface_coord = np.concatenate(
        [
            reference_receptor.coord[receptor_backbone_interface_mask],
            reference_ligand.coord[ligand_backbone_interface_mask],
        ],
        axis=-2,
    )
    # Use these coordinates for superimposition and RMSD calculation
    superimposed_interface_coord, _ = struc.superimpose(
        reference_interface_coord, pose_interface_coord
    )
    rmsd = struc.rmsd(reference_interface_coord, superimposed_interface_coord)
    return rmsd.item() if np.isscalar(rmsd) else rmsd  # type: ignore[union-attr]


def fnat(
    reference_receptor: struc.AtomArray,
    reference_ligand: struc.AtomArray,
    pose_receptor: struc.AtomArray | struc.AtomArrayStack,
    pose_ligand: struc.AtomArray | struc.AtomArrayStack,
    as_peptide: bool = False,
) -> tuple[float | NDArray[np.floating], float | NDArray[np.floating]]:
    """
    Compute the *fnat* and *fnonnat* part of the DockQ score.

    Parameters
    ----------
    reference_receptor, reference_ligand : AtomArray
        The reference receptor and ligand.
    pose_receptor, pose_ligand : AtomArray or AtomArrayStack
        The pose receptor and ligand.
    as_peptide : bool
        If set to true, the chains are treated as CAPRI peptides.

    Returns
    -------
    fnat : float or ndarray, dtype=float
        The percentage of reference contacts that are also found in the pose.
    fnonnat : float or ndarray, dtype=float
        The percentage of pose contacts that are not found in the reference structure.
    """
    cutoff = 4.0 if as_peptide else 5.0

    reference_contacts = _as_set(
        get_contact_residues(reference_receptor, reference_ligand, cutoff)
    )
    if len(reference_contacts) == 0:
        # if there're no contacts between the two chains, fnat and fnonnat are not defined
        nan_values = (
            np.full(shape=len(pose_ligand), fill_value=np.nan)
            if isinstance(pose_ligand, struc.AtomArrayStack)
            else np.nan
        )
        return nan_values, nan_values

    if isinstance(pose_receptor, struc.AtomArray):
        return _calc_fnat_single_model(
            pose_receptor,
            pose_ligand,
            reference_contacts,
            cutoff,
        )
    else:
        fnat = []
        fnonnat = []
        # Multiple poses in an AtomArrayStack -> calculate fnat for each pose
        for receptor, ligand in zip(pose_receptor, pose_ligand):
            fnat_single, fnonnat_single = _calc_fnat_single_model(
                receptor,
                ligand,
                reference_contacts,
                cutoff,
            )
            fnat.append(fnat_single)
            fnonnat.append(fnonnat_single)
        return np.array(fnat, dtype=float), np.array(fnonnat, dtype=float)


def get_contact_residues(
    receptor: struc.AtomArray,
    ligand: struc.AtomArray,
    cutoff: float,
    receptor_mask: NDArray[np.int_] | None = None,
    ligand_mask: NDArray[np.int_] | None = None,
) -> NDArray[np.int_]:
    """
    Get a set of tuples containing the residue IDs for each contact between
    receptor and ligand.

    Parameters
    ----------
    receptor, ligand : AtomArray, shape=(p,)
        The receptor.
    ligand : AtomArray, shape=(q,)
        The ligand.
    cutoff : float
        The distance cutoff for contact.
    receptor_mask : ndarray, shape=(p,), dtype=bool, optional
        A mask that is `True` for atoms in `receptor` that should be considered.
        If `None`, all atoms are considered.
    ligand_mask : ndarray, shape=(q,), dtype=bool, optional
        A mask that is `True` for atoms in `ligand` that should be considered.
        If `None`, all atoms are considered.

    Returns
    -------
    ndarray, shape=(n,2), dtype=int
        Each row represents a contact between receptor and ligand.
        The first column contains the starting atom index of the receptor residue,
        the second column contains the starting atom index of the ligand residue.
    """
    # Put the receptor instead of the ligand into the cell list
    # as the receptor is usually larger
    # This increases the performance in CellList.get_atoms() and _to_sparse_indices()
    cell_list = struc.CellList(receptor, cutoff, selection=receptor_mask)
    if ligand_mask is None:
        all_contacts = cell_list.get_atoms(ligand.coord, cutoff)
    else:
        filtered_contacts = cell_list.get_atoms(ligand.coord[ligand_mask], cutoff)
        all_contacts = np.full(
            (len(ligand), filtered_contacts.shape[-1]),
            -1,
            dtype=filtered_contacts.dtype,
        )
        all_contacts[ligand_mask] = filtered_contacts
    atom_indices = _to_sparse_indices(all_contacts)

    residue_starts = np.stack(
        [
            struc.get_residue_starts_for(receptor, atom_indices[:, 0]),
            struc.get_residue_starts_for(ligand, atom_indices[:, 1]),
        ],
        axis=1,
    )

    # Some contacts might exist between different atoms in the same residues
    return np.unique(residue_starts, axis=0)


def _calc_fnat_single_model(
    receptor: struc.AtomArray,
    ligand: struc.AtomArray,
    reference_contacts: set[tuple[int, int]],
    cutoff: float,
) -> tuple[float | NDArray[np.floating], float | NDArray[np.floating]]:
    """
    Compute the *fnat* and *fnonnat* for a single model.

    Parameters
    ----------
    receptor, ligand : AtomArray
        The pose receptor and ligand.
    reference_contacts : ndarray, shape=(n,2), dtype=int
        The reference contacts.
    cutoff : float
        The distance cutoff for contact.

    Returns
    -------
    fnat : float
        The percentage of reference contacts that are also found in the pose.
    fnonnat : float
        The percentage of pose contacts that are not found in the reference structure.
    """
    pose_contacts = _as_set(get_contact_residues(receptor, ligand, cutoff))
    n_pose = len(pose_contacts)
    n_reference = len(reference_contacts)
    n_true_positive = len(pose_contacts & reference_contacts)
    n_false_positive = len(pose_contacts - reference_contacts)

    if n_reference == 0:
        # Deviation from original DockQ implementation, which returns 0 in this case
        # However, this is misleading as the score is simply not properly defined
        # in this case, as the structure is not a complex
        raise NoContactError("The reference chains do not have any contacts")
    fnat = n_true_positive / n_reference
    fnonnat = n_false_positive / n_pose if n_pose != 0 else 0
    return fnat, fnonnat


def _mask_either_or(
    atoms: struc.AtomArray, atom_name: str, alt_atom_name: str
) -> NDArray[np.bool_]:
    """
    Create a mask that is `True` for all `atom_name` atoms and as fallback
    `alt_atom_name` atoms for all residues that miss `atom_name`.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,)
        The atoms to create the mask for.
    atom_name : str
        The atom name to base the mask on.
    alt_atom_name : str
        The fallback atom name.

    Returns
    -------
    mask : ndarray, shape=(n,), dtype=bool
        The created mask.
    """
    atom_names = atoms.atom_name
    mask = np.zeros(atoms.array_length(), dtype=bool)
    residue_starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    for i in range(len(residue_starts) - 1):
        res_start = residue_starts[i]
        res_stop = residue_starts[i + 1]
        atom_index = np.where(atom_names[res_start:res_stop] == atom_name)[0]
        if len(atom_index) != 0:
            mask[res_start + atom_index] = True
        else:
            # No `atom_name` in residue -> fall back to `alt_atom_name`
            atom_index = np.where(atom_names[res_start:res_stop] == alt_atom_name)[0]
            if len(atom_index) != 0:
                mask[res_start + atom_index] = True
    return mask


def _as_set(array: NDArray[np.int_]) -> set[tuple[int, int]]:
    """
    Convert an array of tuples into a set of tuples.
    """
    return set([tuple(c) for c in array])


def _to_sparse_indices(all_contacts: NDArray[np.int_]) -> NDArray[np.int_]:
    """
    Create tuples of indices that would mark the non-zero elements in a dense
    contact matrix.
    """
    # Find rows where a ligand atom has at least one contact
    non_empty_indices = np.where(np.any(all_contacts != -1, axis=1))[0]
    # Take those rows and flatten them
    receptor_indices = all_contacts[non_empty_indices].flatten()
    # For each row the corresponding ligand atom is the same
    # Hence in the flattened form the ligand atom index is simply repeated
    ligand_indices = np.repeat(non_empty_indices, all_contacts.shape[1])
    combined_indices = np.stack([receptor_indices, ligand_indices], axis=1)
    # Remove the padding values
    return combined_indices[receptor_indices != -1]


@overload
def _scale(rmsd: float, scaling_factor: float) -> float: ...
@overload
def _scale(
    rmsd: NDArray[np.floating], scaling_factor: float
) -> NDArray[np.floating]: ...
def _scale(
    rmsd: float | NDArray[np.floating], scaling_factor: float
) -> float | NDArray[np.floating]:
    """
    Apply the DockQ scaling formula.
    """
    return 1 / (1 + (rmsd / scaling_factor) ** 2)
