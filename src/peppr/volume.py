__all__ = ["volume", "volume_overlap"]

import biotite.structure as struc
import biotite.structure.info as info
import numpy as np
from numpy.typing import NDArray


def volume(molecule: struc.AtomArray, voxel_size: float = 0.5) -> float:
    """
    Calculate the volume of the given molecule.

    Parameters
    ----------
    molecule : struc.AtomArray
        The molecule to calculate the volume of.
    voxel_size : float, optional
        The size of the voxels used for volume calculation.
        The computation becomes more accurate with smaller voxel sizes, but
        the run time scales inverse cubically with voxel size.

    Returns
    -------
    float
        The volume of the molecule.
    """
    voxel_volume = voxel_size**3
    vdw_radii = np.array(
        [info.vdw_radius_single(element) for element in molecule.element]
    )
    max_vdw_radius = np.max(vdw_radii)
    voxel_grid = _create_voxel_grid(molecule, voxel_size, max_vdw_radius)
    cell_list = struc.CellList(voxel_grid, max_vdw_radius)
    occupied_mask = _find_occupied_voxels(cell_list, voxel_grid, molecule, vdw_radii)
    return voxel_volume * np.count_nonzero(occupied_mask)


def volume_overlap(
    molecules: list[struc.AtomArray],
    voxel_size: float = 0.5,
) -> tuple[NDArray[np.floating], float, float]:
    """
    Calculate the volume of the given molecules and how their volumes overlap
    (i.e. their intersection and union).

    Parameters
    ----------
    molecules : list of struc.AtomArray, length=n
        The molecules to calculate the volume of.
    voxel_size : float, optional
        The size of the voxels used for volume calculation.
        The computation becomes more accurate with smaller voxel sizes, but
        the run time scales inverse cubically with voxel size.

    Returns
    -------
    volumes : np.ndarray, shape=(n,), dtype=float
        The volume of each input molecule.
    intersection_volume : float
        The volume intersection of all molecules.
    union_volume : float
        The volume union of all molecules.
    """
    voxel_volume = voxel_size**3

    vdw_radii = [
        np.array([info.vdw_radius_single(element) for element in molecule.element])
        for molecule in molecules
    ]
    max_vdw_radius = max(np.max(radii) for radii in vdw_radii)

    voxel_grids = [
        _create_voxel_grid(molecule, voxel_size, max_vdw_radius)
        for molecule in molecules
    ]
    common_voxel_grid = np.unique(np.concatenate(voxel_grids, axis=0), axis=0)
    cell_list = struc.CellList(common_voxel_grid, max_vdw_radius)
    occupied_masks = [
        _find_occupied_voxels(cell_list, common_voxel_grid, molecule, radii)
        for molecule, radii in zip(molecules, vdw_radii)
    ]
    volumes = [
        voxel_volume * np.count_nonzero(occupied_mask)
        for occupied_mask in occupied_masks
    ]
    intersection_volume = voxel_volume * np.count_nonzero(
        np.all(occupied_masks, axis=0)
    )
    union_volume = voxel_volume * np.count_nonzero(np.any(occupied_masks, axis=0))

    return np.array(volumes), intersection_volume, union_volume


def _create_voxel_grid(
    molecule: struc.AtomArray, voxel_size: float, padding: float
) -> NDArray[np.floating]:
    """
    Create a voxel grid for the given molecule.

    Parameters
    ----------
    molecule : struc.AtomArray
        The molecule to create the voxel grid around.
    voxel_size : float
        The size of the voxels.
    padding : float
        The 'buffer' around the molecule.
        This makes sure that voxels span not only the molecule centers, but also
        their radii.

    Returns
    -------
    voxel_grid : np.ndarray, shape=(n, 3)
        The voxel coordinates.
    """
    min_coord = np.min(molecule.coord, axis=0) - padding
    # Round to the nearest multiple of the voxel size
    # so multiple grids can be combined later using `np.unique()`
    min_voxel_coord = min_coord - min_coord % voxel_size
    max_coord = np.max(molecule.coord, axis=0) + padding
    return np.stack(
        [
            ordinate.ravel()
            for ordinate in np.meshgrid(
                *[
                    np.arange(min_c, max_c, voxel_size)
                    for min_c, max_c in zip(min_voxel_coord, max_coord)
                ]
            )
        ],
        axis=-1,
    )


def _find_occupied_voxels(
    cell_list: struc.CellList,
    voxel_grid: NDArray[np.floating],
    molecule: struc.AtomArray,
    vdw_radii: NDArray[np.floating],
) -> NDArray[np.bool_]:
    """
    Find the voxels that are occupied by the given molecule.

    Parameters
    ----------
    cell_list : struc.CellList
        The cell list that contains the voxel grid.
    voxel_grid : np.ndarray, shape=(n, 3)
        The voxel grid.
    molecule : struc.AtomArray, shape=(k,)
        The molecule to find the occupied voxels for.
    vdw_radii : np.ndarray, shape=(k,), dtype=float
        The van der Waals radii of each atom.

    Returns
    -------
    occupied_voxels : np.ndarray, shape=(n,), dtype=bool
        As mask that is True for each occupied voxel in `voxel_grid`.
    """
    # These voxels are potentially occupied by the molecule...
    atom_indices, voxel_indices = _to_sparse_indices(
        cell_list.get_atoms_in_cells(molecule.coord)
    ).T
    # ... but we need to check if they are within the VdW radius to be sure
    # Use the squared distance to avoid costly square root calculation
    occupied_pair_mask = (
        _squared_distance(molecule.coord[atom_indices], voxel_grid[voxel_indices])
        < vdw_radii[atom_indices] ** 2
    )
    occupied_voxel_indices = voxel_indices[occupied_pair_mask]
    occupied_voxel_mask = np.zeros(voxel_grid.shape[0], dtype=bool)
    occupied_voxel_mask[occupied_voxel_indices] = True
    return occupied_voxel_mask


def _to_sparse_indices(all_contacts: NDArray[np.int_]) -> NDArray[np.int_]:
    """
    Create tuples indices from the :meth:`CellList.get_atoms_in_cells()` return value.

    In other words, they would mark the non-zero elements in a dense contact matrix.

    Parameters
    ----------
    all_contacts : ndarray, dtype=int, shape=(m,n)
        The contact indices as returned by :meth:`CellList.get_atoms_in_cells()`.
        Padded with -1, in the second dimension.
        Dimension *m* marks the query, dimension *n* marks the voxels in the cell list.

    Returns
    -------
    combined_indices : ndarray, dtype=int, shape=(l,2)
        The contact indices.
        Each column contains the query and contact voxel index.
    """
    # Find rows where a query atom has at least one contact
    non_empty_indices = np.where(np.any(all_contacts != -1, axis=1))[0]
    # Take those rows and flatten them
    contact_indices = all_contacts[non_empty_indices].flatten()
    # For each row the corresponding query atom is the same
    # Hence in the flattened form the query atom index is simply repeated
    query_indices = np.repeat(non_empty_indices, all_contacts.shape[1])
    combined_indices = np.stack([query_indices, contact_indices], axis=1)
    # Remove the padding values
    return combined_indices[contact_indices != -1]


def _squared_distance(
    a: NDArray[np.floating], b: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Compute the squared distance between two points.

    Parameters
    ----------
    a, b : np.ndarray, shape=(n, 3)
        The points.

    Returns
    -------
    distance : float
        The squared distance between the two points.
    """
    return np.sum((a - b) ** 2, axis=1)
