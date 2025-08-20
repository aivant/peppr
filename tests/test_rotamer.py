from pathlib import Path
import biotite.structure.io as strucio
import numpy as np
import pytest
from biotite import structure as struc
from peppr.rotamer import (
    ConformerClass,
    RamaResidueType,
    RamaScore,
    RotamerGridResidueMap,
    RotamerScore,
    _check_rama,
    _check_rotamer,
    _get_residue_chis,
    _get_residue_phi_psi_omega,
    _interp_wrapped,
)


@pytest.fixture
def input_pose() -> struc.AtomArray:
    """
    Make an atom array for testing.
    """
    pose = strucio.load_structure(Path(__file__).parent / "data" / "pdb" / "1a3n.cif")
    return pose


@pytest.fixture
def fake_2d_chi_angle_grid() -> struc.AtomArray:
    """
    Create a fake 2D grid for testing.
    This grid is used to test the interpolation function with wrapping.
    """
    # A simple 4x4 grid with values increasing from 0 to 6
    grid = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ]
    )
    # Axis centers for χ1 and χ2
    chi1_centers = np.array([0.0, 90.0, 180.0, 270.0])
    chi2_centers = np.array([0.0, 90.0, 180.0, 270.0])
    wrap = [True, True]

    return {
        "grid": grid,
        "axes": [chi1_centers, chi2_centers],
        "wrap": wrap,
        "steps": [90, 90],
    }


@pytest.fixture
def fake_2d_phi_psi_angle_grid() -> struc.AtomArray:
    """
    Create a fake 2D grid for testing.
    This grid is used to test the interpolation function with wrapping.
    """
    # A simple 4x4 grid with values increasing from 0 to 6
    grid = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ]
    )
    # Axis centers for φ and ψ
    phi_centers = np.array([-179.0, -90.0, 0.0, 90.0])
    psi_centers = np.array([-179.0, -90.0, 0.0, 90.0])
    wrap = [True, True]

    return {
        "grid": grid,
        "axes": [phi_centers, psi_centers],
        "wrap": wrap,
        "steps": [90, 90],
    }


def test_interp_wrapped_basic(fake_2d_chi_angle_grid, fake_2d_phi_psi_angle_grid):
    """Test the interpolation with wrapping on a simple 2D grid.
    This is like finding a value on a 2D contour grid that corresponds
    to axes values which are angles (could be χ1/χ2 or phi/psi angles).

    The test checks:
    1. Exact grid point interpolation.
        The value at (90°, 180°) should be 2.0. Which is at row 1, col 2 of the grid.
    2. Wrapping behavior for chi angles (e.g., 450° should wrap to 90°).
        Sometimes, angles are out of range. The function should wrap these angles correctly.
        Wrapping for chi angles uses the same logic for all chi angles in a coordinate except the last one,
        which is residue-specific, e.g., for ASP, it wraps around 180° while LEU wraps around 360°.
        So, (450°, 180°) for ASP should give the same value as (90°, 0°) while (450°, 180°) for LEU
        should give the same value as (90°, 180°).
    3. Midpoint interpolation between grid points.
        Most times, we want to find a value that is not exactly on the grid.
        For example, if we want to find the value at (45°, 45°),
        it should be should be 1.0, the average of the values at (0°, 0°), (0°, 90°), (90°, 0°), and (90°, 90°).
    4. Wrapping behavior for phi/psi angles to exact value on grid (e.g (270°, -360°) should wrap to (-90°, 0°)).
    5. Midpoint interpolation for phi/psi angles (e.g (-135°, -135°)).
    6. Case where after wrapping, angles falls outside the grid span but within the wrap range.
        For example, for phi/psi, (-190°, 200°) should wrap to (170°, -160°).
        The function should still be able to interpolate the value correctly

    The expected results are calculated based on the known values at the grid points.
    """
    chis_grid_obj = fake_2d_chi_angle_grid
    phi_psi_grid_obj = fake_2d_phi_psi_angle_grid
    chi_grid = chis_grid_obj["grid"]

    # Case 1: exact grid point (90°, 180°)
    val_exact, _ = _interp_wrapped("leu", chis_grid_obj, [90.0, 180.0], "chi")
    expected_exact = chi_grid[1, 2]  # row 1, col 2
    assert np.isclose(val_exact, expected_exact), (
        f"Expected {expected_exact}, got {val_exact}"
    )

    # Case 2:
    # a. wrapping test: For LEU, 450° for χ1 wraps to 90°, 180 for χ1 remains the same
    # and should be wrapped to (90°, 180°)
    val_wrapped, _ = _interp_wrapped("leu", chis_grid_obj, [450.0, 180.0], "chi")
    assert np.isclose(val_wrapped, expected_exact), (
        f"Expected {expected_exact} for wrapped input, got {val_wrapped}"
    )
    # b. wrapping test: For ASP, 450° for χ1 wraps to 90°, 180 for χ2 wraps to 0°
    # and should be wrapped to (90°, 0°)
    # Note: ASP wraps χ2 around 180°
    expected_exact_1 = chi_grid[1, 0]
    val_wrapped, _ = _interp_wrapped("asp", chis_grid_obj, [450.0, 180.0], "chi")
    assert np.isclose(val_wrapped, expected_exact_1), (
        f"Expected {expected_exact_1} for wrapped input, got {val_wrapped}"
    )
    # Case 3: midpoint interpolation: (45°, 45°)
    # Between (0,0) = 0.0, (0,90) = 1.0, (90,0) = 1.0, (90,90) = 2.0
    # Bilinear interpolation should give 1.0
    val_mid, _ = _interp_wrapped("leu", chis_grid_obj, [45.0, 45.0], "chi")
    assert np.isclose(val_mid, 1.0), f"Expected 1.0, got {val_mid}"

    # Case 4: wrapping for phi/psi angles
    # For phi/psi, 270° should wrap to -90° and -360° should wrap to 0°
    # So, (270°, -360°) should wrap to (-90°, 0°)
    # The value at (-90°, 0°) is at row 2, col 2 of the grid
    val_phi_psi, _ = _interp_wrapped("leu", phi_psi_grid_obj, [270, -360], "phi-psi")
    expected_phi_psi = phi_psi_grid_obj["grid"][1, 2]  # row 1, col 2
    assert np.isclose(val_phi_psi, expected_phi_psi), (
        f"Expected {expected_phi_psi}, got {val_phi_psi}"
    )

    # Case 5: Midpoint interpolation for phi/psi: (-135°, -135°)
    # Between (-179,-179)=0.0, (-179,-90)=1.0, (-90,-179)=1.0, (-90,-90)=2.0
    val_mid_phi_psi, _ = _interp_wrapped(
        "leu", phi_psi_grid_obj, [-135.0, -135.0], "phi-psi"
    )
    assert np.isclose(val_mid_phi_psi, 1.0, 0.02), (
        f"Expected 1.0, got {val_mid_phi_psi}"
    )

    # Case 6: Case where after wrapping, angles falls outside the grid span but within the wrap range
    # For phi/psi, (-190°, 200°) should wrap to (170°, -160°); since 170° is outside the grid span,
    # but within the wrap range, it truncates to the nearest grid point, which is 90°. Therefore, we expect
    # the interpolation to be done at (90°, -160°) which is between (90°, -179°)=3.0 and (90°, -90°)=4.0
    # Bilinear interpolation should give 3.2
    val_outside_span, _ = _interp_wrapped(
        "leu", phi_psi_grid_obj, [-190, 200], "phi-psi"
    )
    expected_outside_span = (
        3.5  # Bilinear interpolation gives values between 3.0 and 4.0
    )
    assert np.isclose(val_outside_span, expected_outside_span, 0.5), (
        f"Expected {expected_outside_span}, got {val_outside_span}"
    )


@pytest.mark.parametrize(
    ["residue_id", "chi1", "chi2"],
    [
        (42, -65.6852, -77.1221),
        (2, -61.3018, 168.0359),
    ],
)
def test_get_residue_chis(input_pose, residue_id, chi1, chi2):
    """
    Test the extraction of chi angles from a residue.
    Compare the chi angles obtained from the input pose with values from mmtbx.

    The values we compare against are obtained by running
    >>> from mmtbx.rotamer import rotamer_eval
    >>> import iotbx.pdb
    >>> evaluator = rotamer_eval.RotamerEval()
    >>> input_cif = "tests/data/pdb/1a3n.cif"
    >>> pdb_inp = iotbx.pdb.input(file_name=input_cif)
    >>> hierarchy = pdb_inp.construct_hierarchy()
    >>> model = hierarchy.models()[0]
    >>> chain = model.chains()[0]
    >>> residue_group = chain.residue_groups[41]
    >>> residue = residue_group.only_atom_group()
    >>> evaluator.chi_angles(residue)
    """
    # Get the chis for a specific residue
    chis = _get_residue_chis(
        input_pose, (input_pose.res_id == residue_id) & (input_pose.chain_id == "A")
    )
    assert np.isclose(chis[0], chi1, atol=1e-1)
    assert np.isclose(chis[1], chi2, atol=1e-1)


@pytest.mark.parametrize(
    ["residue_id", "phi", "psi", "omega"],
    [
        (42, -85.6, -5.0, -177.3),
        (25, -63.6, -43.8, 179.9),
        (2, -89.6, 124.7, 176.9),
    ],
)
def test_get_residue_phi_psi_omega(input_pose, residue_id, phi, psi, omega):
    """
    Test the extraction of phi, psi, and omega angles from a residue.
    Compare the angles obtained from the input pose with values from mmtbx.

    The values we compare against are obtained by running
    >>> from mmtbx.validation.ramalyze import ramalyze
    >>> import iotbx.pdb
    >>> input_cif = "tests/data/pdb/1a3n.cif"
    >>> pdb_inp = iotbx.pdb.input(file_name=input_cif)
    >>> hierarchy = pdb_inp.construct_hierarchy()
    >>> rama_eval = ramalyze.ramalyze(pdb_hierarchy=hierarchy)
    >>> for result in rama_eval.results:
    ...     print((result.resseq, result.phi, result.psi)
    """
    # Get the angles for a specific residue
    atoms = input_pose[
        np.isin(input_pose.res_id, [residue_id - 1, residue_id, residue_id + 1])
    ]
    angles = _get_residue_phi_psi_omega(atoms)
    assert np.isclose(angles[0][0], phi, atol=1e-1)
    assert np.isclose(angles[1][0], psi, atol=1e-1)


@pytest.mark.parametrize(
    [
        "phi",
        "psi",
        "resname_tag",
        "pct",
        "classification",
    ],
    [
        (-85.6, -5.0, RamaResidueType.GENERAL, 0.5932, ConformerClass.FAVORED),
        (-63.6, -43.8, RamaResidueType.GLY, 0.97, ConformerClass.FAVORED),
        (-89.6, 124.7, RamaResidueType.GENERAL, 0.3482, ConformerClass.FAVORED),
        (-75.2, 62.9, RamaResidueType.GENERAL, 0.0126, ConformerClass.ALLOWED),
    ],
)
def test_check_rama(phi, psi, resname_tag, pct, classification):
    """
    Test the check_rama function with known values.
    The values we compare against are obtained by running mmtbx validation

    >>> from mmtbx.validation.ramalyze import ramalyze
    >>> import iotbx.pdb
    >>> input_cif = "tests/data/pdb/1a3n.cif"
    >>> pdb_inp = iotbx.pdb.input(file_name=input_cif)
    >>> hierarchy = pdb_inp.construct_hierarchy()
    >>> rama_eval = ramalyze.ramalyze(pdb_hierarchy=hierarchy)
    >>> for result in rama_eval.results:
    ...     print(
    ...         (result.phi, result.psi, <omega-dummy>, result.resseq,
    ...          result.resname.strip().upper(), result.chain_id.strip(),
    ...           result.model_no))
    """
    # Test known values
    result = _check_rama(
        phi=phi,
        psi=psi,
        resname_tag=resname_tag,
    )
    assert result.classification == classification
    assert result.pct == pytest.approx(pct, abs=1e-2)


def test_rama_score_from_atoms(input_pose):
    """
    Test the RamaScore.from_atoms method.
    This method should return a RamaScore object with the expected properties.
    """

    rama_score = RamaScore.from_atoms(input_pose)
    assert isinstance(rama_score, RamaScore)
    assert len(rama_score.rama_scores) > 0
    assert 0.0 <= rama_score.rama_scores[0].pct <= 1


@pytest.mark.parametrize(
    [
        "chi_angles",
        "residue_id",
        "chain_id",
        "pct",
        "classification",
    ],
    [
        (
            [67.36847217387181, 95.16059204939675],
            146,
            "D",
            0.3301,
            ConformerClass.FAVORED,
        ),
        # This is a slight difference in the value from mmtbx with wrapping (282.8778752930479)
        # Since the chi2 grid only goes up to 180, we wrap it to 180.
        ([294.31480801517387, 102.87788887], 42, "A", 0.878, ConformerClass.FAVORED),
        # This is a slight difference in the value from mmtbx with wrapping (323.85937238168424)
        # Since the chi2 grid only goes up to 180, we wrap it to 180.
        ([291.8228091231218, 143.85939955], 6, "A", 0.7489, ConformerClass.FAVORED),
    ],
)
def test_check_rotamer(
    input_pose,
    chi_angles,
    residue_id,
    chain_id,
    pct,
    classification,
):
    """
    Test the check_rotamer function with known values.
    The values we compare against are obtained by running

    >>> from mmtbx.validation.rotalyze import rotalyze
    >>> import iotbx.pdb
    >>> input_cif = "tests/data/pdb/1a3n.cif"
    >>> pdb_inp = iotbx.pdb.input(file_name=input_cif)
    >>> hierarchy = pdb_inp.construct_hierarchy()
    >>> rota_evaluator = rotalyze(pdb_hierarchy=hierarchy)
    >>> for result in rota_evaluator.results:
    ...     print(
    ...         (result.chi_angles, int(result.resid.strip()),
    ...          result.resname.strip().upper(), result.chain_id.strip(), result.score, result.evaluation.upper()
    ...           result.model_no))
    """
    result = _check_rotamer(atom_array=input_pose, res_id=residue_id, chain_id=chain_id)
    score, angles = result
    assert np.isclose(angles[0], chi_angles[0], atol=1e-2)
    assert np.isclose(angles[1], chi_angles[1], atol=1e-2)
    assert np.isclose(score.pct, pct, atol=1e-2)
    assert score.classification == classification


def test_rotamer_score_from_atoms(input_pose):
    """
    Test the RotamerScore.from_atoms method.
    This method should return a RotamerScore object with the expected properties.
    """
    rotamer_score = RotamerScore.from_atoms(input_pose)
    assert isinstance(rotamer_score, RotamerScore)
    assert len(rotamer_score.rotamer_scores) > 0
    assert 0.0 <= rotamer_score.rotamer_scores[0].pct <= 1