__all__ = ["idealize_bonds"]

import biotite.interface.rdkit as rdkit_interface
import biotite.structure as struc
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from peppr.sanitize import sanitize


def idealize_bonds(pose: struc.AtomArray) -> struc.AtomArray:
    """
    Idealize the bonds of a pose.

    This function uses RDKit's MMFF94 force field to idealize the bond
    lengths and angles of a pose. It is intended to make ideal reference
    bond lengths and angles for a pose only. It ignores clashes!

    Parameters
    ----------
    pose : AtomArray
        The structure to idealize.

    Returns
    -------
    AtomArray
        The idealized structure.
    """
    # Generate an rdkit mol
    mol = rdkit_interface.to_mol(pose, explicit_hydrogen=False)
    sanitize(mol)

    # Set `nonBondedThresh` very high and `ignoreInterfragInteractions=True`
    # to effectively ignore clashes
    MMFFOptimizeMolecule(
        mol=mol,
        mmffVariant="MMFF94",
        maxIters=50,
        nonBondedThresh=100.0,
        confId=-1,
        ignoreInterfragInteractions=True,
    )

    # Convert the optimized reference back to an AtomArray
    pose_idealized = rdkit_interface.from_mol(mol, add_hydrogen=False)[0]

    return pose_idealized
