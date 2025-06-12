__all__ = ["is_small_molecule", "standardize"]


import biotite.structure as struc

DONOR_PATTERN = (
    "["
    "$([Nv3!H0,Nv4!H0+1,nH1]),"
    # Guanidine can be tautomeric - e.g. Arginine
    "$([NX3,NX2]([!O,!S])!@C(!@[NX3,NX2]([!O,!S]))!@[NX3,NX2]([!O,!S])),"
    "$([O,S;!H0])"
    "]"
)
ACCEPTOR_PATTERN = (
    "["
    # Oxygens and Sulfurs
    # singly protonotated can be acceptors
    "$([O,S;v2H1]),"
    # O,S that is unprotonotated, neutral or negative (but not part of nitro-like group!)
    "$([O,S;v2H0;!$([O,S]=N-*)]),"
    "$([O,S;-;!$(*-N=[O,S])]),"
    # also include neutral aromatic oxygen and sulfur
    "$([s,o;+0]),"
    # Nitrogens
    # aromatic unprotonated nitrogens (not trivalent connectivity?)
    "$([nH0+0;!X3]),"
    # nitrile
    "$([ND1H0;$(N#[Cv4])]),"
    # unprotonated nitrogen next to aromatic ring
    "$([Nv3H0;$(N-c)]),"
    # Fluorine on aromatic ring, only
    "$([F;$(F-[#6]);!$(FC[F,Cl,Br,I])])"
    "]"
)
HALOGEN_PATTERN = "[F,Cl,Br,I;+0]"
HBOND_DISTANCE_SCALING = (0.8, 1.15)
HALOGEN_DISTANCE_SCALING = (0.8, 1.05)


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
