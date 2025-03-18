import biotite.structure.info as info
import numpy as np
import pytest
import peppr


@pytest.mark.parametrize("comp_id", ["ALA", "TRP", "BNZ"])
def test_find_clashes(comp_id):
    """
    Within molecules from the CCD, :func:`find_clashes` should not find any clashes.
    However, if a second copy of the same molecule with the same coordinates are
    added to the system, at least each atom clashes with the corresponding atom in the
    other molecule."
    """
    molecule1 = info.residue(comp_id)
    molecule1 = molecule1[molecule1.element != "H"]
    molecule2 = molecule1.copy()
    molecule1.res_id[:] = 1
    molecule2.res_id[:] = 2
    system = molecule1 + molecule2

    clashes = peppr.find_clashes(system)

    # No clashes within the same molecule
    assert not np.any(system.res_id[clashes[:, 0]] == system.res_id[clashes[:, 1]])
    # At least each atom clashes with the corresponding atom in the other molecule
    corresponding_atom_clash_mask = (
        system.res_id[clashes[:, 0]] != system.res_id[clashes[:, 1]]
    ) & (system.atom_name[clashes[:, 0]] == system.atom_name[clashes[:, 1]])
    assert np.count_nonzero(corresponding_atom_clash_mask) == molecule1.array_length()
