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

    If multiple models were used to calculate *DockQ*, the attributes are arrays.

    Attributes
    ----------
    fnat : float or ndarray, dtype=float
        The fraction of native contacts found in the model relative to the total
        number of native contacts.
    fnonnat : float or ndarray, dtype=float
        The fraction of non-native contacts found in the model relative to the total
        number of model contacts.
    irmsd : float or ndarray, dtype=float
        The interface RMSD.
    lrmsd : float or ndarray, dtype=float
        The ligand RMSD.
    score : float or ndarray, dtype=float
        The DockQ score.
    n_models : int or None
        The number of models for which the *DockQ* was calculated.
        `None`, if the *DockQ* was calculated for an `AtomArray`.
    model_receptor_index, model_ligand_index, native_receptor_index, native_ligand_index : int or None
        The indices of the model and native chain that were included for *DockQ*
        computation.
        Only set, if called from `global_dockq()`.
    """

    fnat: float | NDArray[np.floating]
    fnonnat: float | NDArray[np.floating]
    irmsd: float | NDArray[np.floating]
    lrmsd: float | NDArray[np.floating]
    score: float | NDArray[np.floating] = field(init=False)
    n_models: int | None = field(init=False)
    model_receptor_index: int | None = None
    model_ligand_index: int | None = None
    native_receptor_index: int | None = None
    native_ligand_index: int | None = None

    def __post_init__(self) -> None:
        score = np.mean(
            [self.fnat, _scale(self.irmsd, 1.5), _scale(self.lrmsd, 8.5)], axis=0
        )
        n_models = None if np.isscalar(score) else len(score)
        super().__setattr__("score", score)
        super().__setattr__("n_models", n_models)

    def for_model(self, model_index: int) -> "DockQ":
        """
        Get the DockQ results for a specific model index.

        Parameters
        ----------
        model_index : int
            The index of the model for which the DockQ results should be retrieved.

        Returns
        -------
        DockQ
            The DockQ results for the specified model index.

        Raises
        ------
        IndexError
            If the `GlobalDockQ` object was computed for a single model,
            i.e. `n_models` is `None`.
        """
        if self.n_models is None:
            raise IndexError("DockQ was computed for a single model")
        return DockQ(
            self.fnat[model_index].item(),  # type: ignore[index]
            self.fnonnat[model_index].item(),  # type: ignore[index]
            self.irmsd[model_index].item(),  # type: ignore[index]
            self.lrmsd[model_index].item(),  # type: ignore[index]
            model_receptor_index=self.model_receptor_index,
            model_ligand_index=self.model_ligand_index,
            native_receptor_index=self.native_receptor_index,
            native_ligand_index=self.native_ligand_index,
        )


def dockq(
    native_receptor: struc.AtomArray,
    native_ligand: struc.AtomArray,
    model_receptor: struc.AtomArray | struc.AtomArrayStack,
    model_ligand: struc.AtomArray | struc.AtomArrayStack,
    as_peptide: bool = False,
) -> DockQ:
    """
    Compute *DockQ* for a single pair of receptor and ligand in both, the model and
    native structure.

    Parameters
    ----------
    native_receptor, native_ligand : AtomArray
        The native receptor and ligand.
    model_receptor, model_ligand : AtomArray or AtomArrayStack
        The model receptor and ligand.
        Multiple models can be provided.
    as_peptide : bool
        If set to true, the chains are treated as CAPRI peptides.

    Returns
    -------
    DockQ
        The DockQ result.
        If multiple models are provided, the `DockQ` attributes are arrays.

    Notes
    -----
    If the ligand is a small molecule, an associated `BondList` is required in
    `model_ligand` and `native_ligand` for mapping the atoms between them.

    Examples
    --------

    Single chains as expected as input.

    >>> model_receptor = model_complex[model_complex.chain_id == "C"]
    >>> model_ligand = model_complex[model_complex.chain_id == "B"]
    >>> native_receptor = native_complex[native_complex.chain_id == "C"]
    >>> native_ligand = native_complex[native_complex.chain_id == "B"]
    >>> dockq_result = dockq(model_receptor, model_ligand, native_receptor, native_ligand)
    >>> print(f"{dockq_result.fnat:.2f}")
    0.50
    >>> print(f"{dockq_result.irmsd:.2f}")
    2.10
    >>> print(f"{dockq_result.lrmsd:.2f}")
    8.13
    >>> print(f"{dockq_result.score:.2f}")
    0.45
    """
    if as_peptide:
        if any(
            [
                is_small_molecule(chain)
                for chain in (
                    model_receptor,
                    model_ligand,
                    native_receptor,
                    native_ligand,
                )
            ]
        ):
            raise ValueError("'as_peptide' is true, but the chains are not peptides")

    if is_small_molecule(model_ligand):
        # For small molecules DockQ is only based on the pocket-aligned ligand RMSD
        lrmsd_ = pocket_aligned_lrmsd(
            native_receptor, native_ligand, model_receptor, model_ligand
        )
        zero = 0 if isinstance(lrmsd_, float) else np.zeros(len(lrmsd_))
        return DockQ(zero, zero, zero, lrmsd_)

    else:
        lrmsd_ = lrmsd(native_receptor, native_ligand, model_receptor, model_ligand)

        fnat_, fnonnat_ = fnat(
            native_receptor,
            native_ligand,
            model_receptor,
            model_ligand,
            as_peptide,
        )

        irmsd_ = irmsd(
            native_receptor,
            native_ligand,
            model_receptor,
            model_ligand,
            as_peptide,
        )

        return DockQ(fnat_, fnonnat_, irmsd_, lrmsd_)


def pocket_aligned_lrmsd(
    native_receptor: struc.AtomArray,
    native_ligand: struc.AtomArray,
    model_receptor: struc.AtomArray | struc.AtomArrayStack,
    model_ligand: struc.AtomArray | struc.AtomArrayStack,
) -> float | NDArray[np.floating]:
    """
    Compute the pocket-aligned RMSD part of the DockQ score for small molecules.

    Parameters
    ----------
    native_receptor, native_ligand : AtomArray
        The native receptor and ligand.
    model_receptor, model_ligand : AtomArray
        The model receptor and ligand.

    Returns
    -------
    float or ndarray, dtype=float
        The pocket-aligned RMSD.
    """
    native_contacts = get_contact_residues(
        native_receptor,
        native_ligand,
        cutoff=10.0,
    )
    if len(native_contacts) == 0:
        # if there're no contacts between the two chains, no lrmsd is calculated
        return (
            np.full(shape=len(model_ligand), fill_value=np.nan)
            if isinstance(model_ligand, struc.AtomArrayStack)
            else np.nan
        )
    # Create mask which is True for all backbone atoms in contact receptor residues
    interface_mask = struc.get_residue_masks(
        native_receptor, native_contacts[:, 0]
    ).any(axis=0) & np.isin(native_receptor.atom_name, BACKBONE_ATOMS)
    # Use interface backbone coordinates for pocket-aligned superimposition
    _, transform = struc.superimpose(
        native_receptor.coord[interface_mask],
        model_receptor.coord[..., interface_mask, :],
    )
    # Use the superimposed coordinates for RMSD calculation between ligand atoms
    lrmsd = struc.rmsd(native_ligand.coord, transform.apply(model_ligand.coord))
    return lrmsd.item() if np.isscalar(lrmsd) else lrmsd  # type: ignore[union-attr]


def lrmsd(
    native_receptor: struc.AtomArray,
    native_ligand: struc.AtomArray,
    model_receptor: struc.AtomArray | struc.AtomArrayStack,
    model_ligand: struc.AtomArray | struc.AtomArrayStack,
) -> float | NDArray[np.floating]:
    """
    Compute the ligand RMSD part of the DockQ score.

    Parameters
    ----------
    native_receptor, native_ligand : AtomArray
        The native receptor and ligand.
    model_receptor, model_ligand : AtomArray
        The model receptor and ligand.

    Returns
    -------
    float or ndarray, dtype=float
        The ligand RMSD.
    """
    receptor_relevant_mask = np.isin(model_receptor.atom_name, BACKBONE_ATOMS)
    if is_small_molecule(model_ligand):
        # For small molecules include all heavy atoms
        ligand_relevant_mask = np.full(model_ligand.array_length(), True)
    else:
        ligand_relevant_mask = np.isin(model_ligand.atom_name, BACKBONE_ATOMS)

    model_receptor_coord = model_receptor.coord[..., receptor_relevant_mask, :]
    model_ligand_coord = model_ligand.coord[..., ligand_relevant_mask, :]
    native_receptor_coord = native_receptor.coord[receptor_relevant_mask]
    native_ligand_coord = native_ligand.coord[ligand_relevant_mask]
    _, transform = struc.superimpose(
        native_receptor_coord,
        model_receptor_coord,
    )
    superimposed_ligand_coord = transform.apply(model_ligand_coord)
    lrmsd = struc.rmsd(native_ligand_coord, superimposed_ligand_coord)
    return lrmsd.item() if np.isscalar(lrmsd) else lrmsd  # type: ignore[union-attr]


def irmsd(
    native_receptor: struc.AtomArray,
    native_ligand: struc.AtomArray,
    model_receptor: struc.AtomArray | struc.AtomArrayStack,
    model_ligand: struc.AtomArray | struc.AtomArrayStack,
    as_peptide: bool = False,
) -> float | NDArray[np.floating]:
    """
    Compute the interface RMSD part of the DockQ score.

    Parameters
    ----------
    native_receptor, native_ligand : AtomArray
        The native receptor and ligand.
    model_receptor, model_ligand : AtomArray or AtomArrayStack
        The model receptor and ligand.
    as_peptide : bool
        If set to true, the chains are treated as CAPRI peptides.

    Returns
    -------
    float or ndarray, dtype=float
        The interface RMSD.
    """
    if as_peptide:
        cutoff = 8.0
        receptor_mask = _mask_either_or(native_receptor, "CB", "CA")
        ligand_mask = _mask_either_or(native_ligand, "CB", "CA")
    else:
        cutoff = 10.0
        receptor_mask = None
        ligand_mask = None
    native_contacts = get_contact_residues(
        native_receptor,
        native_ligand,
        cutoff,
        receptor_mask,
        ligand_mask,
    )

    if len(native_contacts) == 0:
        # if there're no contacts between the two chains,
        # no irmsd is calculated
        return (
            np.full(shape=len(model_ligand), fill_value=np.nan)
            if isinstance(model_ligand, struc.AtomArrayStack)
            else np.nan
        )

    # Create mask which is True for all backbone atoms in contact residues
    receptor_backbone_interface_mask = struc.get_residue_masks(
        native_receptor, native_contacts[:, 0]
    ).any(axis=0) & np.isin(native_receptor.atom_name, BACKBONE_ATOMS)
    ligand_backbone_interface_mask = struc.get_residue_masks(
        native_ligand, native_contacts[:, 1]
    ).any(axis=0) & np.isin(native_ligand.atom_name, BACKBONE_ATOMS)

    # Get the coordinates of interface backbone atoms
    model_interface_coord = np.concatenate(
        [
            model_receptor.coord[..., receptor_backbone_interface_mask, :],
            model_ligand.coord[..., ligand_backbone_interface_mask, :],
        ],
        axis=-2,
    )
    native_interface_coord = np.concatenate(
        [
            native_receptor.coord[receptor_backbone_interface_mask],
            native_ligand.coord[ligand_backbone_interface_mask],
        ],
        axis=-2,
    )
    # Use these coordinates for superimposition and RMSD calculation
    superimposed_interface_coord, _ = struc.superimpose(
        native_interface_coord, model_interface_coord
    )
    rmsd = struc.rmsd(native_interface_coord, superimposed_interface_coord)
    return rmsd.item() if np.isscalar(rmsd) else rmsd  # type: ignore[union-attr]


def fnat(
    native_receptor: struc.AtomArray,
    native_ligand: struc.AtomArray,
    model_receptor: struc.AtomArray | struc.AtomArrayStack,
    model_ligand: struc.AtomArray | struc.AtomArrayStack,
    as_peptide: bool = False,
) -> tuple[float | NDArray[np.floating], float | NDArray[np.floating]]:
    """
    Compute the *fnat* and *fnonnat* part of the DockQ score.

    Parameters
    ----------
    native_receptor, native_ligand : AtomArray
        The native receptor and ligand.
    model_receptor, model_ligand : AtomArray or AtomArrayStack
        The model receptor and ligand.
    as_peptide : bool
        If set to true, the chains are treated as CAPRI peptides.

    Returns
    -------
    fnat : float or ndarray, dtype=float
        The percentage of native contacts that are also found in the model.
    fnonnat : float or ndarray, dtype=float
        The percentage of model contacts that are not found in the native structure.
    """
    cutoff = 4.0 if as_peptide else 5.0

    native_contacts = _as_set(
        get_contact_residues(native_receptor, native_ligand, cutoff)
    )
    if len(native_contacts) == 0:
        # if there're no contacts between the two chains, fnat and fnonnat are not defined
        nan_values = (
            np.full(shape=len(model_ligand), fill_value=np.nan)
            if isinstance(model_ligand, struc.AtomArrayStack)
            else np.nan
        )
        return nan_values, nan_values

    if isinstance(model_receptor, struc.AtomArray):
        return _calc_fnat_single_model(
            model_receptor,
            model_ligand,
            native_contacts,
            cutoff,
        )
    else:
        fnat = []
        fnonnat = []
        # Multiple models in an AtomArrayStack -> calculate fnat for each model
        for receptor, ligand in zip(model_receptor, model_ligand):
            fnat_single, fnonnat_single = _calc_fnat_single_model(
                receptor,
                ligand,
                native_contacts,
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
    native_contacts: set[tuple[int, int]],
    cutoff: float,
) -> tuple[float | NDArray[np.floating], float | NDArray[np.floating]]:
    """
    Compute the *fnat* and *fnonnat* for a single model.

    Parameters
    ----------
    receptor, ligand : AtomArray
        The model receptor and ligand.
    native_contacts : ndarray, shape=(n,2), dtype=int
        The native contacts.
    cutoff : float
        The distance cutoff for contact.

    Returns
    -------
    fnat : float
        The percentage of native contacts that are also found in the model.
    fnonnat : float
        The percentage of model contacts that are not found in the native structure.
    """
    model_contacts = _as_set(get_contact_residues(receptor, ligand, cutoff))
    n_model = len(model_contacts)
    n_native = len(native_contacts)
    n_true_positive = len(model_contacts & native_contacts)
    n_false_positive = len(model_contacts - native_contacts)

    if n_native == 0:
        # Deviation from original DockQ implementation, which returns 0 in this case
        # However, this is misleading as the score is simply not properly defined
        # in this case, as the structure is not a complex
        raise NoContactError("The native chains do not have any contacts")
    fnat = n_true_positive / n_native
    fnonnat = n_false_positive / n_model if n_model != 0 else 0
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
