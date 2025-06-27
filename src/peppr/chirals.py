__all__ = ["get_chirality"]


from typing import Dict
from rdkit import Chem
from rdkit.Chem import AllChem


def get_chirality(mol: Chem.Mol) -> Dict[int, str]:
    """
    Extract chirality information from an RDKit molecule.

    This function identifies all tetrahedral chiral centers in the molecule
    and returns their chirality in CW/CCW notation.

    Parameters
    ----------
    mol : Chem.Mol
        The RDKit molecule to analyze.

    Returns
    -------
    Dict[int, str]
        A dictionary mapping atom indices to chirality labels.
        Keys are atom indices (0-based), values are either 'CW' or 'CCW'.
        Only chiral centers are included in the dictionary.

    Notes
    -----
    The function uses RDKit's native chirality representation in CW/CCW format.
    Only atoms that are actually chiral centers will be included in the result.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('C[C@H](N)C(=O)O')  # L-alanine
    >>> chirality = get_chirality(mol)
    >>> print(chirality)
    {1: 'CCW'}
    """
    if mol is None:
        raise ValueError("Invalid molecule provided")

    # Ensure the molecule has 3D coordinates for proper chirality perception
    if mol.GetNumConformers() == 0:
        # Generate 3D coordinates if none exist
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)

    chirality_dict = {}

    for atom in mol.GetAtoms():
        # Check if this atom is a chiral center
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            # Get the chirality in CW/CCW notation (RDKit's native format)
            chiral_tag = atom.GetChiralTag()

            if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                chirality_dict[atom.GetIdx()] = 'CW'
            elif chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                chirality_dict[atom.GetIdx()] = 'CCW'

    return chirality_dict



