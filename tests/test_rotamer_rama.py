from pathlib import Path
import math
from unittest import result
from peppr.rotamer_rama import (
    get_residue_chis,
    get_residue_phi_psi_omega,
    check_rama,
    check_rotamer,
    RamaScore,
    interp_wrapped,
)
from biotite import structure as struc
import biotite.structure.io as strucio
import numpy as np
import pytest


@pytest.fixture
def input_pose() -> struc.AtomArray:
    """
    Make an atom array for testing.
    """
    pose = strucio.load_structure(Path(__file__).parent / "data" / "pdb" / "1a3n.cif")
    return pose


@pytest.fixture
def fake_2d_grid() -> struc.AtomArray:
    """
    Create a fake 2D grid for testing.
    """
    # Define a simple 2D grid: value = row_index + col_index
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
        "span": [(0, 360), (0, 360)],
    }


def test_interp_wrapped_basic(fake_2d_grid):
    grid_obj = fake_2d_grid
    grid = grid_obj["grid"]

    # Case 1: exact grid point (90°, 180°)
    val_exact, _ = interp_wrapped(grid_obj, [90.0, 180.0])
    expected_exact = grid[1, 2]  # row 1, col 2
    assert np.isclose(val_exact, expected_exact), (
        f"Expected {expected_exact}, got {val_exact}"
    )

    # Case 2: wrapping test: 450° for χ1 wraps to 90°, should give same as (90°, 180°)
    val_wrapped, _ = interp_wrapped(grid_obj, [450.0, 180.0])
    assert np.isclose(val_wrapped, expected_exact), (
        f"Expected {expected_exact} for wrapped input, got {val_wrapped}"
    )

    # Case 3: midpoint interpolation: (45°, 45°)
    # Between (0,0) = 0.0, (0,90) = 1.0, (90,0) = 1.0, (90,90) = 2.0
    # Bilinear interpolation should give 1.0
    val_mid, _ = interp_wrapped(grid_obj, [45.0, 45.0])
    assert np.isclose(val_mid, 1.0), f"Expected 1.0, got {val_mid}"


@pytest.mark.parametrize(
    ["residue_id", "chi1", "chi2"],
    [
        # test some edge cases
        # Obtained by running  mmtbx.validation.ramalyze.
        # >>> from mmtbx.rotamer import rotamer_eval
        # >>> import iotbx.pdb
        # >>> evaluator = rotamer_eval.RotamerEval()
        # >>> input_cif = "tests/data/pdb/1a3n.cif"
        # >>> pdb_inp = iotbx.pdb.input(file_name=input_cif)
        # >>> hierarchy = pdb_inp.construct_hierarchy()
        # >>> model = hierarchy.models()[0]
        # >>> chain = model.chains()[0]
        # >>> residue_group = chain.residue_groups[41]
        # >>> residue = residue_group.only_atom_group()
        # >>> evaluator.chi_angles(residue)
        (42, -65.6852, -77.1221),
        (2, -61.3018, 168.0359),
    ],
)
def test_get_residue_chis(input_pose, residue_id, chi1, chi2):
    """Test the extraction of chi angles from a residue."""
    # Get the chis for a specific residue
    chis = get_residue_chis(
        input_pose, (input_pose.res_id == residue_id) & (input_pose.chain_id == "A")
    )

    assert np.isclose(chis["chi1"], chi1, atol=1e-1)
    assert np.isclose(chis["chi2"], chi2, atol=1e-1)


@pytest.mark.parametrize(
    ["residue_id", "phi", "psi", "omega"],
    [
        # test some edge cases
        # Obtained by running  mmtbx.validation.ramalyze.
        # >>> from mmtbx.validation.ramalyze import ramalyze
        # >>> import iotbx.pdb
        # >>> input_cif = "tests/data/pdb/1a3n.cif"
        # >>> pdb_inp = iotbx.pdb.input(file_name=input_cif)
        # >>> hierarchy = pdb_inp.construct_hierarchy()
        # >>> rama_eval = ramalyze.ramalyze(pdb_hierarchy=hierarchy)
        # >>> for result in rama_eval.results:
        # ...     print((result.resseq, result.phi, result.psi)
        (42, -85.6, -5.0, -177.3),
        (25, -63.6, -43.8, 179.9),
        (2, -89.6, 124.7, 176.9),
    ],
)
def test_get_residue_phi_psi_omega(input_pose, residue_id, phi, psi, omega):
    """Test the extraction of phi, psi, and omega angles from a residue."""
    # Get the angles for a specific residue
    atoms = input_pose[
        np.isin(input_pose.res_id, [residue_id - 1, residue_id, residue_id + 1])
    ]
    angles = get_residue_phi_psi_omega(atoms)
    assert np.isclose(angles["phi"][0], phi, atol=1e-1)
    assert np.isclose(angles["psi"][0], psi, atol=1e-1)


@pytest.mark.parametrize(
    [
        "phi",
        "psi",
        "omega",
        "residue_id",
        "resname",
        "chain_id",
        "resname_tag",
        "model_no",
        "rama_score_pct",
        "classification",
    ],
    [
        # test some edge cases
        # Obtained by running  mmtbx.validation.ramalyze.
        # >>> from mmtbx.validation.ramalyze import ramalyze
        # >>> import iotbx.pdb
        # >>> input_cif = "tests/data/pdb/1a3n.cif"
        # >>> pdb_inp = iotbx.pdb.input(file_name=input_cif)
        # >>> hierarchy = pdb_inp.construct_hierarchy()
        # >>> rama_eval = ramalyze.ramalyze(pdb_hierarchy=hierarchy)
        # >>> for result in rama_eval.results:
        # ...     print(
        # ...         (result.phi, result.psi, <omega-dummy>, result.resseq,
        # ...          result.resname.strip().upper(), result.chain_id.strip(),
        # ...           result.model_no))
        #  omega angle and resname_tag are not from mmtbx, added post-facto
        (-85.6, -5.0, -177.3, 42, "TRY", "A", "general", 0, 0.5932, "FAVORED"),
        (-63.6, -43.8, 179.9, 25, "GLY", "A", "gly", 0, 0.97, "FAVORED"),
        (-89.6, 124.7, 176.9, 2, "LEU", "A", "general", 0, 0.3482, "FAVORED"),
        (-75.2, 62.9, 180.0, 117, "PHE", "A", "general", 0, 0.0126, "ALLOWED"),
    ],
)
def test_check_rama(
    phi,
    psi,
    omega,
    residue_id,
    resname,
    chain_id,
    resname_tag,
    model_no,
    rama_score_pct,
    classification,
):
    # Test known values
    result = check_rama(
        phi=phi,
        psi=psi,
        omega=omega,
        res_id=residue_id,
        resname=resname,
        chain_id=chain_id,
        resname_tag=resname_tag,
        model_no=model_no,
    )
    assert result["classification"] == classification
    assert result["rama_score_pct"] == pytest.approx(rama_score_pct, abs=1e-2)


def test_rama_score_from_atom_array(input_pose):
    rama_score = RamaScore.from_atom_array(input_pose)
    assert isinstance(rama_score, RamaScore)
    assert len(rama_score.rama_scores) > 0
    assert rama_score.rama_scores[0].classification in ["FAVORED", "ALLOWED", "OUTLIER"]
    assert rama_score.rama_scores[0].rama_score_pct > 0.0
    assert rama_score.rama_scores[0].resname_tag in [
        "general",
        "gly",
        "cispro",
        "prepro",
        "transpro",
        "ileval",
    ]


@pytest.mark.parametrize(
    [
        "chi_angles",
        "residue_id",
        "resname",
        "chain_id",
        "model_no",
        "rotamer_score_pct",
        "classification",
    ],
    [
        # test some edge cases
        # Obtained by running  mmtbx.validation.ramalyze.
        # >>> from mmtbx.validation.rotalyze import rotalyze
        # >>> import iotbx.pdb
        # >>> input_cif = "tests/data/pdb/1a3n.cif"
        # >>> pdb_inp = iotbx.pdb.input(file_name=input_cif)
        # >>> hierarchy = pdb_inp.construct_hierarchy()
        # >>> rota_evaluator = rotalyze(pdb_hierarchy=hierarchy)
        # >>> for result in rota_evaluator.results:
        # ...     print(
        # ...         (result.chi_angles, int(result.resid.strip()),
        # ...          result.resname.strip().upper(), result.chain_id.strip(), result.score, result.evaluation.upper()
        # ...           result.model_no))
        ([67.36847217387181, 95.16059204939675], 146, "HIS", "D", 0, 0.3301, "FAVORED"),
        # This is a slight difference in the value from mmtbx with wrapping (282.8778752930479)
        # Since the chi2 grid only goes up to 180, we wrap it to 180.
        ([294.31480801517387, 102.87788887], 42, "TYR", "A", 0, 0.878, "FAVORED"),
        # This is a slight difference in the value from mmtbx with wrapping (323.85937238168424)
        # Since the chi2 grid only goes up to 180, we wrap it to 180.
        ([291.8228091231218, 143.85939955], 6, "ASP", "A", 0, 0.7489, "FAVORED"),
    ],
)
def test_check_rotamer(
    input_pose,
    chi_angles,
    residue_id,
    chain_id,
    resname,
    model_no,
    rotamer_score_pct,
    classification,
):
    result = check_rotamer(
        atom_array=input_pose, res_id=residue_id, chain_id=chain_id, model_no=model_no
    )

    assert result["resname"] == resname
    assert result["resid"] == residue_id
    assert np.isclose(result["observed"]["chi1"], chi_angles[0], atol=1e-2)
    assert np.isclose(result["observed"]["chi2"], chi_angles[1], atol=1e-2)
    assert np.isclose(result["rotamer_score_pct"], rotamer_score_pct, atol=1e-2)
    assert result["classification"] == classification
