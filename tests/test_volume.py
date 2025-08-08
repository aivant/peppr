import biotite.structure as struc
import biotite.structure.info as info
import numpy as np
import pytest
import peppr


def test_sphere_volume():
    """
    Check if the discretized volume of an atom is close to the volume of its VdW sphere,
    if the voxel size is small enough.
    """
    ELEMENT = "C"

    molecule = struc.AtomArray(1)
    molecule.element[:] = ELEMENT
    # The atom position should not matter
    molecule.coord = np.random.default_rng(seed=0).uniform(-10, 10, size=(1, 3))

    ref_volume = 4 / 3 * np.pi * info.vdw_radius_single(ELEMENT) ** 3

    test_volume = peppr.volume(molecule, voxel_size=0.1)

    assert test_volume == pytest.approx(ref_volume, rel=1e-2)


# Use elements with significantly different VdW radii
@pytest.mark.parametrize("elements", [("C", "C"), ("C", "RB")])
@pytest.mark.parametrize("seed", range(5))
def test_combined_volume(elements, seed):
    """
    A molecule consisting of two atoms should have the volume of the VdW sphere of the
    larger atom, when both atoms are at the same position.
    When moving both atoms apart the total volume should monotonically increase until
    the the total volume is equal to the sum of both atom volumes.
    """
    MAX_DISPLACEMENT = 5
    DISPLACEMENT_STEPS = 20
    # Sometimes the curve is not monotonically increasing,
    # due to inaccuracies of using voxels
    MONOTONOUS_TOLERANCE = 1e-1

    molecule = struc.AtomArray(2)
    molecule.element = elements
    # The atom positions should not matter
    molecule.coord = np.random.default_rng(seed=0).uniform(-10, 10, size=(2, 3))

    vdw_radii = np.array([info.vdw_radius_single(element) for element in elements])
    sphere_volumes = 4 / 3 * np.pi * vdw_radii**3

    rng = np.random.default_rng(seed=seed)
    displacement_direction = rng.uniform(-1, 1, size=3)
    displacement_direction /= np.linalg.norm(displacement_direction)
    displacements = (
        displacement_direction
        * np.linspace(0, MAX_DISPLACEMENT, DISPLACEMENT_STEPS)[:, None]
    )

    volumes = []
    for displacement in displacements:
        molecule.coord[1] = molecule.coord[0] + displacement
        volumes.append(peppr.volume(molecule, 0.1))

    assert np.all(np.diff(volumes) >= -MONOTONOUS_TOLERANCE), (
        "Volumes should be monotonically increasing"
    )

    assert volumes[0] == pytest.approx(np.max(sphere_volumes), rel=1e-2), (
        "Initial volume should be equal to the larger sphere volume"
    )
    assert volumes[-1] == pytest.approx(np.sum(sphere_volumes), rel=1e-2), (
        "Final volume should be equal to the sum of the sphere volumes"
    )


def test_volume_consistency():
    """
    :func:`volume()` should return the same volume as :func:`volume_overlap()`
    for the same molecules.
    """
    COMP_NAMES = ["GLY", "TRP", "BTN"]

    molecules = [info.residue(comp_name) for comp_name in COMP_NAMES]
    molecules = [molecule[molecule.element != "H"] for molecule in molecules]

    ref_volumes = [peppr.volume(molecule) for molecule in molecules]
    test_volumes, _, _ = peppr.volume_overlap(molecules)

    assert np.allclose(ref_volumes, test_volumes.tolist())


@pytest.mark.parametrize("displace_slightly", [False, True])
@pytest.mark.parametrize("comp_name", ["GLY", "TRP", "BTN"])
def test_volume_overlap(comp_name, displace_slightly):
    """
    For any molecule and its exact copy, the volume union and intersection should be
    equal to the volume of the molecule, if they are perfectly occluding each other.
    If they are sufficiently far away, the volume union should be twice the molecule
    volume, and the volume intersection should be zero.

    Also test a very slight displacement of the molecule, to check if the
    common voxel grid is calculated correctly, i.e. duplicate voxels are removed.
    """
    SLIGHT_DISPLACEMENT = 1e-3
    FAR_DISPLACEMENT = 100

    pose = info.residue(comp_name)
    pose = pose[pose.element != "H"]
    same_pose = pose.copy()
    if displace_slightly:
        same_pose.coord += SLIGHT_DISPLACEMENT
    different_pose = same_pose.copy()
    different_pose.coord += FAR_DISPLACEMENT

    # Case of occluding molecules
    volumes, intersection_volume, union_volume = peppr.volume_overlap([pose, same_pose])
    molecule_volume = volumes[0].item()
    # Both molecules are identical
    # -> volumes should be equal (with some inaccuracies due to using voxels)
    assert volumes[1] == pytest.approx(molecule_volume, rel=1e-2)
    assert intersection_volume == pytest.approx(molecule_volume, rel=1e-2)
    assert union_volume == pytest.approx(molecule_volume, rel=1e-2)

    # Case of far away molecules
    volumes, intersection_volume, union_volume = peppr.volume_overlap(
        [pose, different_pose]
    )
    molecule_volume = volumes[0].item()
    assert volumes[1] == pytest.approx(molecule_volume, rel=1e-2)
    assert intersection_volume == 0.0
    assert union_volume == pytest.approx(2 * molecule_volume, rel=1e-2)
