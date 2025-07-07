import itertools
import biotite.structure as struc
import numpy as np
import pytest
import peppr
from tests.common import (
    assemble_predictions,
    get_reference_metric,
    list_test_predictions,
)

ALL_METRICS = [
    peppr.MonomerRMSD(5.0, ca_only=True),
    peppr.MonomerRMSD(5.0, ca_only=False),
    peppr.MonomerTMScore(),
    peppr.MonomerLDDTScore(),
    peppr.IntraLigandLDDTScore(),
    peppr.LDDTPLIScore(),
    peppr.LDDTPPIScore(),
    peppr.GlobalLDDTScore(backbone_only=True),
    peppr.GlobalLDDTScore(backbone_only=False),
    peppr.DockQScore(include_pli=False),
    peppr.DockQScore(include_pli=True),
    peppr.LigandRMSD(),
    peppr.InterfaceRMSD(),
    peppr.ContactFraction(),
    peppr.PocketAlignedLigandRMSD(),
    peppr.BiSyRMSD(5.0),
    peppr.BondLengthViolations(),
    peppr.ClashCount(),
    peppr.BondAngleViolations(),
    peppr.ChiralityViolations(),
]


@pytest.mark.parametrize(
    ["metric", "value_range"],
    [
        (peppr.MonomerRMSD(5.0, ca_only=True), (0.0, 10.0)),
        (peppr.MonomerRMSD(5.0, ca_only=False), (0.0, 10.0)),
        (peppr.MonomerTMScore(), (0.4, 1.0)),
        (peppr.MonomerLDDTScore(), (0.4, 1.0)),
        (peppr.IntraLigandLDDTScore(), (0.4, 1.0)),
        (peppr.LDDTPLIScore(), (0.0, 1.0)),
        (peppr.LDDTPPIScore(), (0.0, 1.0)),
        (peppr.GlobalLDDTScore(backbone_only=True), (0.0, 1.0)),
        (peppr.GlobalLDDTScore(backbone_only=False), (0.0, 1.0)),
        (peppr.DockQScore(include_pli=False), (0.0, 1.0)),
        (peppr.DockQScore(include_pli=True), (0.2, 1.0)),
        # Upper end is quite high, but validated with DockQ reference implementation
        (peppr.LigandRMSD(), (0.0, 80.0)),
        (peppr.InterfaceRMSD(), (0.0, 30.0)),
        (peppr.ContactFraction(), (0.0, 1.0)),
        (peppr.PocketAlignedLigandRMSD(), (0.0, 30.0)),
        (peppr.BiSyRMSD(5.0), (0.0, 40.0)),
        (peppr.BondLengthViolations(), (0.0, 1.0)),
        (peppr.ClashCount(), (0, 10000)),
        (peppr.BondAngleViolations(), (0.0, 1.0)),
        (peppr.ChiralityViolations(), (0.0, 1.0)),
    ],
    ids=lambda x: x.name if isinstance(x, peppr.Metric) else "",
)
def test_metrics(metric, value_range):
    """
    Check for each implemented metric simply whether :meth:`Metric.evaluate()` works
    without error and returns an array with correct shape with values in a reasonable
    range.
    """
    SYSTEM_ID = "7znt__2__1.F_1.G__1.J"

    reference, poses = assemble_predictions(SYSTEM_ID)
    for pose in poses:
        reference_order, pose_order = peppr.find_optimal_match(reference, pose)
        reference = reference[reference_order]
        pose = pose[pose_order]
        value = metric.evaluate(reference, pose)
        assert value >= value_range[0]
        assert value <= value_range[1]


@pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda metric: metric.name)
def test_no_modification(metric):
    """
    No metric should modify the input structures.
    """
    # Choose any system that is suitable for all metrics, i.e. contains PPI and PLI
    reference, poses = assemble_predictions("7znt__2__1.F_1.G__1.J")
    pose = poses[0]
    original_reference = reference.copy()
    original_pose = pose.copy()
    metric.evaluate(reference, pose)
    assert np.all(reference == original_reference)
    assert np.all(pose == original_pose)


@pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda metric: metric.name)
def test_unsuitable_system(metric):
    """
    If a system is unsuitable for the metric, *NaN* should be returned.
    There should never be an exception raised.
    To check this, run all metrics on an empty system, which is never suitable.
    """
    reference = struc.AtomArray(0)
    pose = struc.AtomArray(0)
    assert np.isnan(metric.evaluate(reference, pose))


def test_unique_names():
    """
    Check if the names of the implemented metrics are unique.
    """
    names = set(metric.name for metric in ALL_METRICS)
    assert len(names) == len(ALL_METRICS)


@pytest.mark.parametrize(
    ["metric", "column_name", "abs_tolerance", "rel_tolerance"],
    [
        pytest.param(
            peppr.LDDTPLIScore(),
            "lddt_pli",
            0.5,
            0.3,
            # Expect failure for now, as peppr follows definition from CASP 15
            # while OpenStructure uses the CASP 16 definition
            marks=pytest.mark.xfail(reason="Different metric definition"),
        ),
        pytest.param(
            peppr.BiSyRMSD(5.0, inclusion_radius=4.0, outlier_distance=np.inf),
            "bisy_rmsd",
            0.25,
            0.04,
            # Slight differences as OpenStructure adopted a different definition
            # than the one used in the paper
            marks=pytest.mark.xfail(reason="Different metric definition"),
        ),
    ],
)
@pytest.mark.parametrize("system_id", list_test_predictions())
def test_pli_metrics(system_id, metric, column_name, abs_tolerance, rel_tolerance):
    """
    Check PLI metrics against reference values from *OpenStructureToolkit*.
    """
    ref_metric = get_reference_metric(system_id, column_name)

    reference, poses = assemble_predictions(system_id)
    evaluator = peppr.Evaluator([metric])
    # The reference computes metrics per ligand,
    # so for comparison we do this here as well
    for ligand_i in range(ref_metric.shape[1]):
        chain_mask = (
            # Either a peptide chain ...
            np.char.isdigit(reference.chain_id)
            # ... or the selected ligand
            | (reference.chain_id == f"LIG{ligand_i}")
        )
        masked_reference = reference[chain_mask]
        masked_poses = poses[:, chain_mask]
        evaluator.feed(system_id, masked_reference, masked_poses)
    test_metric = np.stack(evaluator.get_results()[0], axis=1)
    test_metric = _find_matching_ligand(ref_metric, test_metric)

    assert test_metric.flatten().tolist() == pytest.approx(
        ref_metric.flatten().tolist(), abs=abs_tolerance, rel=rel_tolerance, nan_ok=True
    )


@pytest.mark.parametrize("move_structure", ["reference", "pose"])
@pytest.mark.parametrize("system_id", list_test_predictions())
def test_intra_ligand_lddt_score(system_id, move_structure):
    """
    In the intra ligand lDDT the score should not depend on the relative position of the
    ligands in the system, as the lDDT is computed per each ligand.
    Hence moving the ligands towards each other in the reference or pose should not
    change the lDDT score.
    """
    reference, poses = assemble_predictions(system_id)
    metric = peppr.IntraLigandLDDTScore()
    # Explicitly do not use the `Evaluator` here,
    # as moving ligand might change the atom mapping
    original_lddt = np.array([metric.evaluate(reference, pose) for pose in poses])

    # Move the ligands (and proteins as well) towards (or rather into) each other
    if move_structure == "reference":
        reference = _place_at_origin(reference)
    elif move_structure == "pose":
        poses = _place_at_origin(poses)
    moved_lddt = np.array([metric.evaluate(reference, pose) for pose in poses])

    assert original_lddt.tolist() == moved_lddt.tolist()


def _find_matching_ligand(ref_metrics, test_metrics):
    """
    The numbering in the ``chain`` column of ``metrics.parquet`` does not correspond to
    the numbering in the ``small_molecule_*.sdf`` files.
    Hence this function reorders the columns (the ligands) of the test metrics to match
    the reference metrics as closely as possible.

    Parameters
    ----------
    ref_metrics, test_metrics : np.ndarray, shape=(n_poses, n_ligands), dtype=float
        The metrics.

    Returns
    -------
    reordered_test_metrics : np.ndarray, shape=(n_poses, n_ligands), dtype=float
        The reordered test metrics that match the reference metrics as closely as
        possible.
    """
    smallest_diff = np.inf
    best_order = None
    for order in itertools.permutations(range(test_metrics.shape[1])):
        diff = np.sum(np.abs(ref_metrics - test_metrics[:, order]))
        if diff < smallest_diff:
            smallest_diff = diff
            best_order = order
    return test_metrics[:, best_order]


def _place_at_origin(system):
    """
    Place each chain (ligands and proteins as well) into the coordinate origin

    Parameters
    ----------
    system : AtomArray or AtomArrayStack
        The system to place into the coordinate origin.

    Returns
    -------
    system : AtomArray or AtomArrayStack
        The system with the chains placed into the coordinate origin.
    """
    chain_starts = struc.get_chain_starts(system, add_exclusive_stop=True)
    for start_i, stop_i in itertools.pairwise(chain_starts):
        centroid = system.coord[..., start_i:stop_i, :].mean(axis=-2)
        system.coord[..., start_i:stop_i, :] -= centroid[..., None, :]
    return system


def test_ligand_rmsd_with_no_contacts():
    """
    LigandRMSD metric should return NaN when the reference only contains two chains and they are not in contact.

    We first load a system, only keep two chains and move the reference ligand away to make sure they reference receptor
    and reference ligand are not in contact. Then we compute the LigandRMSD and check if it
    is NaN.

    :param data_dir: pytest fixture, path to the directory containing the test data
    """
    reference, poses = assemble_predictions("7znt__2__1.F_1.G__1.J")
    pose = poses[0]
    reference_order, pose_order = peppr.find_optimal_match(reference, pose)
    reference = reference[reference_order]
    pose = pose[pose_order]

    # Assert that the reference contains only two protein chains
    assert set(reference.chain_id[~reference.hetero]) == {"0", "1"}

    # Assert that LigandRMSD is not NaN before translation
    metric = peppr.LigandRMSD()
    lrmsd = metric.evaluate(reference, pose)
    assert not np.isnan(lrmsd)

    # Traslate chain 0 of the reference so that it is not in contact with chain 1
    reference[reference.chain_id == "0"] = struc.translate(
        reference[reference.chain_id == "0"], [200, 200, 200]
    )

    # Assert that LigandRMSD is NaN after translation
    lrmsd = metric.evaluate(reference, pose)
    assert np.isnan(lrmsd)


@pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda metric: metric.name)
def test_ligand_only_system(metric):
    """
    Check that metrics work correctly when given only ligand atoms.
    This tests that metrics don't make assumptions about the presence of protein chains
    that could break when working with ligand-only systems.
    """
    # Choose any system that is suitable for all metrics
    reference, poses = assemble_predictions("7znt__2__1.F_1.G__1.J")
    pose = poses[0]

    # Keep only ligand atoms
    reference = reference[reference.chain_id == "LIG0"]
    pose = pose[pose.chain_id == "LIG0"]

    # The metric should still work without raising an exception
    value = metric.evaluate(reference, pose)

    # For most metrics, we expect a valid numeric value or NaN
    # (NaN is acceptable for metrics that can't handle the input)
    assert np.isnan(value) or isinstance(value, (int, float))


def test_chirality_violations():
    """
    Comparing a molecule with itself should return 0% chirality violations.
    Comparing a molecule with its mirror image should return 100% chirality violations.
    """
    SYSTEM_ID = "7znt__2__1.F_1.G__1.J"
    reference, _ = assemble_predictions(SYSTEM_ID)

    metric = peppr.ChiralityViolations()

    # Compare the molecule with itself
    assert metric.evaluate(reference, reference) == 0.0

    # Compare the molecule with its mirror image
    mirror_image = reference.copy()
    mirror_image.coord[:, 0] *= -1
    assert metric.evaluate(reference, mirror_image) == 1.0
