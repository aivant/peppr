"""Tests for chirality functions."""

import pytest
from rdkit import Chem
from peppr.chirals import get_chirality


@pytest.mark.parametrize(
    "smiles,expected",
    [
        # Ethane: no chiral centers
        ("CC", {}),
        # L-alanine: one chiral center at index 1, should be CCW
        ("C[C@H](N)C(=O)O", {1: "CCW"}),
        # D-alanine: one chiral center at index 1, should be CW
        ("C[C@@H](N)C(=O)O", {1: "CW"}),
        # (2R,3S)-2,3-dihydroxybutane: two chiral centers, both present
        ("C[C@H](O)[C@H](O)C", {1: "CCW", 3: "CCW"}),
    ],
)
def test_get_chirality(smiles, expected):
    """Test get_chirality for various molecules and expected chiralities."""
    mol = Chem.MolFromSmiles(smiles)
    result = get_chirality(mol)
    assert result == expected
