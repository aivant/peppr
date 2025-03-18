__all__ = ["find_clashes"]

import biotite.structure as struc
import biotite.structure.info as info
import numpy as np
from numpy.typing import NDArray


def find_clashes(atoms: struc.AtomArray, vdw_scaling: float = 0.65) -> NDArray[np.int_]:
    """
    Find atom clashes in the given structure.

    A clash is defined as a pair of non-bonded atoms whose distance is smaller than
    the sum of their Van-der-Waals radii (multiplied by a scaling factor).

    Parameters
    ----------
    atoms : AtomArray
        The structure to find the clashes in.
    vdw_scaling : float, optional
        The scaling factor for the Van-der-Waals radii.

    Returns
    -------
    ndarray, shape=(n,2), dtype=int
        The array of clashes.
        Each element represents a pair of atom indices that are in clash.
    """
    if atoms.array_length() == 0:
        return np.zeros((0, 2), dtype=int)

    # Although we only consider heavy atoms,
    # we cannot properly use ProtOr radii as they lead to intra residue clashes
    vdw_radii = np.array([info.vdw_radius_single(e) for e in atoms.element])

    # Any pair of atoms, whose distance is larger than the largest possible VdW radius,
    # is not relevant for clash detection
    max_clash_threshold = 2 * vdw_scaling * np.max(vdw_radii)
    contacts = _find_contacts(atoms, max_clash_threshold)
    distances = struc.index_distance(atoms, contacts)
    clash_thresholds = vdw_scaling * (
        vdw_radii[contacts[:, 0]] + vdw_radii[contacts[:, 1]]
    )
    clashes = contacts[distances < clash_thresholds]

    # Atoms that are connected by a bond are not considered clashes
    bond_set = set([frozenset((i, j)) for i, j, _ in atoms.bonds.as_array()])
    clash_set = set([frozenset((i, j)) for i, j in clashes])
    clash_set -= bond_set
    # Ensure two dimensional array, even if there are no clashes
    return np.array([tuple(e) for e in clash_set], dtype=int).reshape(-1, 2)


def _find_contacts(
    atoms: struc.AtomArray,
    inclusion_radius: float,
) -> NDArray[np.int_]:
    """
    Find contacts between the atoms in the given structure.

    Parameters
    ----------
    atoms : AtomArray
        The structure to find the contacts for.
    inclusion_radius : float
        Pairwise atom distances are considered within this radius.

    Returns
    -------
    ndarray, shape=(n,2), dtype=int
        The array of contacts.
        Each element represents a pair of atom indices that are in contact.
    """
    coords = struc.coord(atoms)
    # Use a cell list to find atoms within inclusion radius in O(n) time complexity
    cell_list = struc.CellList(coords, inclusion_radius)
    # Pairs of indices for atoms within the inclusion radius
    all_contacts = cell_list.get_atoms(coords, inclusion_radius)
    # Convert into pairs of indices
    contacts = _to_sparse_indices(all_contacts)
    # Do not consider self-contacts
    contacts = contacts[contacts[:, 0] != contacts[:, 1]]
    return contacts


def _to_sparse_indices(all_contacts: NDArray[np.int_]) -> NDArray[np.int_]:
    """
    Create tuples of contact indices from the :meth:`CellList.get_atoms()` return value.

    In other words, they would mark the non-zero elements in a dense contact matrix.

    Parameters
    ----------
    all_contacts : ndarray, dtype=int, shape=(m,n)
        The contact indices as returned by :meth:`CellList.get_atoms()`.
        Padded with -1, in the second dimension.
        Dimension *m* marks the query atoms, dimension *n* marks the contact atoms.

    Returns
    -------
    ndarray, dtype=int, shape=(l,2)
        The contact indices.
        Each column contains the query and contact atom index.
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
