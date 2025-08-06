__all__ = ["estimate_formal_charges"]

import functools
from pathlib import Path
import biotite.interface.rdkit as rdkit_interface
import biotite.structure as struc
import numpy as np
import pandas as pd
import rdkit.Chem.AllChem as Chem
from numpy.typing import NDArray
from peppr.sanitize import sanitize

METAL_CHARGES = {
    # For metals with multiple oxidation states, the smallest charge is assumed
    # For the computation of salt bridges it is only important that an atom has a
    # positive charge, anyway
    "LI": 1,
    "K": 1,
    "NA": 1,
    "RB": 1,
    "BE": 2,
    "MG": 2,
    "CA": 2,
    "CR": 2,
    "MO": 3,
    "MN": 2,
    "FE": 2,
    "RU": 3,
    "CO": 2,
    "NI": 2,
    "CU": 1,
    "ZN": 2,
    "AL": 3,
}


def estimate_formal_charges(
    atoms: struc.AtomArray, ph: float = 7.4
) -> NDArray[np.int_]:
    """
    Determine the formal charge of each atom in the structure.

    Parameters
    ----------
    atoms : AtomArray
        The atoms to determine the formal charge for.
        Must have an associated :class:`BondList`.
    ph : float, optional
        The pH of the environment.
        By default a physiological pH value is used [1]_.

    Returns
    -------
    np.ndarray, shape=(n,)
        The formal charge of each atom.

    References
    ----------
    .. [1] https://www.ncbi.nlm.nih.gov/books/NBK507807/
    """

    patterns = []
    for row in _get_protonation_table().itertuples():
        if _is_active(ph, row.ph):
            patterns.append((_get_pattern(row.group), float(row.charge)))
        else:
            # neutralize when inactive!
            patterns.append((_get_pattern(row.group), 0.0))

    mol = rdkit_interface.to_mol(atoms)
    sanitize(mol)
    # Initialize with the original charges
    charges = rdkit_interface.from_mol(mol, add_hydrogen=False).charge
    # Remove hydrogen atoms to correctly match SMARTS atoms with the '[D<n>]' property
    mol = Chem.RemoveAllHs(mol, sanitize=False)  # type: ignore[attr-defined]

    for pattern, charge in patterns:
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            if len(match) == 1:
                # There is only one matched atom -> assign charge to it
                atom_index = match[0]
            elif (index := _get_atom_index_for_map(pattern, 1)) is not None:
                # Alternatively, use the atom with map number 1 ('[X:1]' in SMARTS)
                atom_index = match[index]
            else:
                raise ValueError(
                    "The pattern does not specify the atom to assign the charge to"
                )
            charges[atom_index] = charge

    # Assign charges to metals
    metal_charges = np.array(
        [METAL_CHARGES.get(element.upper(), 0) for element in atoms.element]
    )
    charges[metal_charges != 0] = metal_charges[metal_charges != 0]

    return charges


@functools.cache
def _get_protonation_table() -> pd.DataFrame:
    """
    Load the table that determines the protonation/deprotonation reactions based on pH.

    Returns
    -------
    table : pd.DataFrame
        The table.
    """
    return pd.read_csv(
        Path(__file__).parent / "protonation.csv",
        delimiter=";",
        dtype=str,
        index_col=False,
        comment="#",
        keep_default_na=False,  # Use literally '' for empty pH condition
    )


@functools.cache
def _get_pattern(group_pattern: str) -> Chem.Mol:
    """
    Create the :class:`Mol` from SMARTS pattern.

    Parameters
    ----------
    group_pattern : str
        The SMARTS pattern.

    Returns
    -------
    mol : Chem.Mol
        The pattern as :class:`Mol`.
    """
    return Chem.MolFromSmarts(group_pattern)  # type: ignore[attr-defined]


def _is_active(ph: float, ph_condition: str) -> bool:
    """
    Check if the protonation/deprotonation should occur at the given pH.

    Parameters
    ----------
    ph : float
        The pH.
    ph_condition : str
        The pH condition, e.g. ``<7.4``.

    Returns
    -------
    reacts : bool
        If the protonation/deprotonation should occur.
        If the pH condition is empty, always return ``True``.
    """
    ph_condition = ph_condition.strip().replace("=", "")
    if ph_condition == "":
        return True
    elif ph_condition.startswith("<"):
        return ph < float(ph_condition[1:])
    elif ph_condition.startswith(">"):
        return ph > float(ph_condition[1:])
    else:
        raise ValueError(f"Invalid pH condition: {ph_condition}")


def _get_atom_index_for_map(pattern: Chem.Mol, map_number: int) -> int | None:
    """
    Get the index of the atom with the given map number.

    Parameters
    ----------
    pattern : Chem.Mol
        The pattern.
    map_number : int
        The map number.

    Returns
    -------
    atom_index : int or None
        The index of the atom with the given map number.
        If no atom with the given map number exists, return ``None``.
    """
    if map_number < 1:
        raise ValueError("Map number must be greater than 0")
    for atom in pattern.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num == map_number:
            return atom.GetIdx()
    return None
