import itertools
import biotite.interface.rdkit as rdkit_interface
import biotite.structure as struc
import numpy as np
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix
from rdkit.rdBase import BlockLogs
from peppr.sanitize import sanitize


def get_distance_bounds(atoms: struc.AtomArray) -> np.ndarray:
    """
    Use RDKit's :func:`GetMoleculeBoundsMatrix()` to get distance bounds for the given
    atoms.

    Run this for every residue separately to increase the performance.

    Parameters
    ----------
    atoms : AtomArray
        The atoms to get the distance bounds for.

    Returns
    -------
    np.ndarray, shape=(n, n)
        The distance bounds matrix.
        The lower bounds are in the lower triangle and the upper bounds are in the
        upper triangle.
        Values across residues are set to NaN.
    """
    bounds_matrix = np.full(
        (atoms.array_length(), atoms.array_length()), np.nan, dtype=float
    )

    for start, stop in itertools.pairwise(
        struc.get_residue_starts(atoms, add_exclusive_stop=True)
    ):
        mol = rdkit_interface.to_mol(atoms[start:stop])
        try:
            sanitize(mol)
        except Exception:
            raise struc.BadStructureError("Failed to sanitize residue")
        with BlockLogs():
            bounds = GetMoleculeBoundsMatrix(mol)
        bounds_matrix[start:stop, start:stop] = bounds

    return bounds_matrix
