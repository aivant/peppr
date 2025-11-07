__all__ = ["MoleculeType", "standardize"]


import functools
from enum import Enum, auto
import biotite.structure as struc
import biotite.structure.info as info

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


class MoleculeType(Enum):
    """
    The type of a molecule.
    """

    SMALL_MOLECULE = auto()
    PROTEIN = auto()
    NUCLEIC_ACID = auto()

    @staticmethod
    def of(chain: struc.AtomArray) -> "MoleculeType":
        """
        Determine the :class:`MoleculeType` of the given chain.

        Parameters
        ----------
        chain : struc.AtomArray, shape=(n,)
            The chain to determine the :class:`MoleculeType` of.

        Returns
        -------
        MoleculeType
            The :class:`MoleculeType` of the given chain.

        Warnings
        --------
        This function does not check if `chain` is truly a single chain, this is the
        responsibility of the caller.
        """
        if chain.hetero[0].item():
            return MoleculeType.SMALL_MOLECULE
        res_name = chain.res_name[0].item()
        if res_name in _amino_acid_names():
            return MoleculeType.PROTEIN
        if res_name in _nucleotide_names():
            return MoleculeType.NUCLEIC_ACID
        raise ValueError(
            f"Chain contains residue '{res_name}' which is not polymer component, "
            "but it is also not marked as a small molecule"
        )


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
    remove_monoatomic_ions: bool = True,
) -> struc.AtomArray | struc.AtomArrayStack:
    """
    Standardize the given system.

    This function

    - removes hydrogen atoms
    - removes solvent atoms
    - removes monoatomic ions, if specified
    - checks if an associated :class:`biotite.structure.BondList` is present

    Parameters
    ----------
    system : struc.AtomArray or struc.AtomArrayStack
        The system to standardize.
    remove_monoatomic_ions : bool, optional
        If set to ``True``, monoatomic ions will be removed from system.

    Returns
    -------
    struc.AtomArray or struc.AtomArrayStack
        The standardized system.
    """
    if system.bonds is None:
        raise ValueError("The system must have an associated BondList")
    mask = (system.element != "H") & ~struc.filter_solvent(system)
    if remove_monoatomic_ions:
        mask &= ~struc.filter_monoatomic_ions(system)
    return system[..., mask]


@functools.cache
def _amino_acid_names() -> set[str]:
    """
    Get the names of all residues considered to be amino acids.
    """
    return set(info.amino_acid_names())


@functools.cache
def _nucleotide_names() -> set[str]:
    """
    Get the names of all residues considered to be nucleotides.
    """
    return set(info.nucleotide_names())
