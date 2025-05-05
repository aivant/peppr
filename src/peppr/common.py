__all__ = ["is_small_molecule", "standardize"]


import biotite.structure as struc


def is_small_molecule(chain: struc.AtomArray) -> bool:
    """
    Check whether the given chain is a small molecule.

    Parameters
    ----------
    chain : struc.AtomArray, shape=(n,)
        The chain to check.

    Returns
    -------
    bool
        Whether the chain is a small molecule.
    """
    return chain.hetero[0].item()


def standardize(
    system: struc.AtomArray | struc.AtomArrayStack,
) -> struc.AtomArray | struc.AtomArrayStack:
    """
    Standardize the given system.

    This function

    - removes hydrogen atoms
    - removes solvent atoms and monoatomic ions
    - checks if an associated :class:`biotite.structure.BondList` is present

    Parameters
    ----------
    system : struc.AtomArray or struc.AtomArrayStack
        The system to standardize.

    Returns
    -------
    struc.AtomArray or struc.AtomArrayStack
        The standardize system.
    """
    if system.bonds is None:
        raise ValueError("The system must have an associated BondList")
    mask = (
        (system.element != "H")
        & ~struc.filter_solvent(system)
        & ~struc.filter_monoatomic_ions(system)
    )
    return system[..., mask]
