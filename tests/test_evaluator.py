import copy
import itertools
from collections import OrderedDict
from pathlib import Path
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest
import peppr
from tests.common import assemble_predictions, list_test_predictions


@pytest.fixture(scope="module")
def metrics():
    return [peppr.MonomerRMSD(5.0), peppr.MonomerLDDTScore(), peppr.MonomerTMScore()]


@pytest.fixture
def selectors():
    return [peppr.MeanSelector(), peppr.OracleSelector()]


@pytest.fixture(scope="module", params=list(peppr.Evaluator.MatchMethod))
def evaluator(metrics, request):
    match_method = request.param
    # Avoid combinatorial explosion for time consuming metrics
    max_matches = 10 if match_method == peppr.Evaluator.MatchMethod.INDIVIDUAL else None
    return peppr.Evaluator(metrics, match_method, max_matches)


@pytest.fixture(scope="module")
def evaluator_fed_with_single_poses(evaluator):
    evaluator = copy.deepcopy(evaluator)
    system_ids = list_test_predictions()
    for system_id in system_ids:
        reference, poses = assemble_predictions(system_id)
        evaluator.feed(system_id, reference, poses[0])
    return evaluator


@pytest.fixture(scope="module")
def evaluator_fed_with_multiple_poses(evaluator):
    evaluator = copy.deepcopy(evaluator)
    system_ids = list_test_predictions()
    for system_id in system_ids:
        reference, poses = assemble_predictions(system_id)
        evaluator.feed(system_id, reference, poses)
    return evaluator


def test_tabulate_metrics_for_single(evaluator_fed_with_single_poses):
    """
    Check if :meth:`Evaluator.tabulate_metrics()` returns a dataframe with the correct
    system IDs and column names, if a single pose is fed for each system.
    """
    table = evaluator_fed_with_single_poses.tabulate_metrics()

    assert table.index.to_list() == list_test_predictions()
    assert table.columns.to_list() == [
        "backbone RMSD",
        "intra polymer lDDT",
        "TM-score",
    ]


def test_tabulate_metrics_for_multi(evaluator_fed_with_multiple_poses, selectors):
    """
    Check if :meth:`Evaluator.tabulate_metrics()` returns a dataframe with the correct
    system IDs and column names, if multiple poses are fed for each system.
    """
    table = evaluator_fed_with_multiple_poses.tabulate_metrics(selectors)

    assert table.index.to_list() == list_test_predictions()
    assert table.columns.to_list() == [
        "backbone RMSD (mean)",
        "backbone RMSD (Oracle)",
        "intra polymer lDDT (mean)",
        "intra polymer lDDT (Oracle)",
        "TM-score (mean)",
        "TM-score (Oracle)",
    ]


@pytest.mark.parametrize("match_method", list(peppr.Evaluator.MatchMethod))
@pytest.mark.parametrize("selectors", [None, [peppr.MeanSelector()]])
def test_tabulate_metrics_with_unsuitable_metric(selectors, match_method):
    """
    Check if metric values appear as *NaN* in the output dataframe, if the metric
    does not work for a given system.
    """

    class UnsuitableMetric(peppr.Metric):
        """
        Placeholder metric that does not work for any system.
        """

        @property
        def name(self):
            return "Unsuitable"

        def evaluate(self, reference, poses):
            return np.nan

        def requires_hydrogen(self):
            return False

        def smaller_is_better(self):
            return False

    evaluator = peppr.Evaluator(
        [peppr.MonomerRMSD(5.0), UnsuitableMetric()], match_method
    )
    system_ids = list_test_predictions()
    for system_id in system_ids:
        reference, poses = assemble_predictions(system_id)
        evaluator.feed(system_id, reference, poses[0])
    table = evaluator.tabulate_metrics(selectors)

    # The first metric (MonomerRMSD) should work
    assert table.iloc[:, 0].notna().all()
    # But the second metric (UnsuitableMetric) should not
    assert table.iloc[:, 1].isna().all()


@pytest.mark.parametrize("single_pose", [True, False])
def test_summarize_metrics(
    single_pose,
    evaluator_fed_with_single_poses,
    evaluator_fed_with_multiple_poses,
    selectors,
):
    """
    Check if :meth:`Evaluator.summarize_metrics()` returns a dictionary that maps
    the expected metric names to floating point values, when fed with a single pose
    per system.
    """
    METRIC_NAMES = [
        "backbone RMSD <5.0",
        "backbone RMSD >5.0",
        "backbone RMSD mean",
        "backbone RMSD median",
        "intra polymer lDDT mean",
        "intra polymer lDDT median",
        "TM-score mean",
        "TM-score median",
        "TM-score random",
        "TM-score ambiguous",
        "TM-score similar",
    ]
    SELECTOR_NAMES = ["mean", "Oracle"]

    if single_pose:
        summary = evaluator_fed_with_single_poses.summarize_metrics()
    else:
        summary = evaluator_fed_with_multiple_poses.summarize_metrics(selectors)

    if single_pose:
        assert set(summary.keys()) == set(METRIC_NAMES)
    else:
        assert set(summary.keys()) == set(
            [
                f"{metric_name} ({selector_name})"
                for metric_name, selector_name in itertools.product(
                    METRIC_NAMES, SELECTOR_NAMES
                )
            ]
        )
    values = np.array(list(summary.values()))
    assert np.issubdtype(values.dtype, float)
    assert np.isfinite(list(summary.values())).all()


@pytest.mark.parametrize("match_method", list(peppr.Evaluator.MatchMethod))
def test_summarize_metrics_with_nan(match_method):
    """
    Check if :meth:`Evaluator.summarize_metrics()` is able to handle metrics that
    return *NaN* for some systems.
    """

    class SometimesUnsuitableMetric(peppr.Metric):
        """
        Placeholder metric that does not work for half of the systems.
        """

        def __init__(self):
            self._rng = np.random.default_rng(seed=0)
            self._works = False
            super().__init__()

        @property
        def name(self):
            return "Metric"

        @property
        def thresholds(self):
            return OrderedDict([("bad", 0.0), ("good", 0.5)])

        def evaluate(self, reference, poses):
            if self._works:
                self._works = False
                return self._rng.random()
            else:
                self._works = True
                return np.nan

        def requires_hydrogen(self):
            return False

        def smaller_is_better(self):
            return False

    evaluator = peppr.Evaluator([SometimesUnsuitableMetric()], match_method)
    system_ids = list_test_predictions()
    for system_id in system_ids:
        reference, poses = assemble_predictions(system_id)
        evaluator.feed(system_id, reference, poses[0])
    summary = evaluator.summarize_metrics()

    # Although the metric is NaN for half of the systems,
    # the summary should still contain finite values, ignoring NaNs
    assert np.isfinite(list(summary.values())).all()
    # Summing the bins should give 1.0, i.e. NaNs are ignored in computing percentages
    assert summary["Metric bad"] + summary["Metric good"] == pytest.approx(1.0)


@pytest.mark.parametrize("tolerate_exceptions", [False, True])
@pytest.mark.parametrize("match_method", list(peppr.Evaluator.MatchMethod))
def test_tolerate_exceptions(match_method, tolerate_exceptions):
    """
    Check if ``tolerate_exceptions=True`` leads to warnings instead of exceptions.
    If ``tolerate_exceptions=False``, the exception should be raised.
    """
    EXCEPTION_MESSAGE = "Expected failure"

    class BrokenMetric(peppr.Metric):
        """
        Placeholder metric that always fails.
        """

        @property
        def name(self):
            return "Broken"

        def evaluate(self, reference, poses):
            raise ValueError(EXCEPTION_MESSAGE)

        def requires_hydrogen(self):
            return False

        def smaller_is_better(self):
            return False

    evaluator = peppr.Evaluator(
        [BrokenMetric()], match_method, tolerate_exceptions=tolerate_exceptions
    )
    system_id = list_test_predictions()[0]
    reference, poses = assemble_predictions(system_id)
    if tolerate_exceptions:
        with pytest.warns(
            peppr.EvaluationWarning,
            match=f"Failed to evaluate Broken on '{system_id}': {EXCEPTION_MESSAGE}",
        ):
            evaluator.feed(system_id, reference, poses[0])
    else:
        with pytest.raises(
            ValueError,
            match=EXCEPTION_MESSAGE,
        ):
            evaluator.feed(system_id, reference, poses[0])


def test_combine():
    """
    Check if an evaluator combined from multiple evaluators fed with one system each
    contains the same results as the evaluator fed with all systems.
    """
    ref_evaluator = peppr.Evaluator([peppr.MonomerRMSD(5.0), peppr.MonomerLDDTScore()])
    system_ids = list_test_predictions()

    split_evaluators = [copy.deepcopy(ref_evaluator) for _ in range(len(system_ids))]
    for split_evaluator, system_id in zip(split_evaluators, system_ids):
        reference, poses = assemble_predictions(system_id)
        split_evaluator.feed(system_id, reference, poses[0])
        ref_evaluator.feed(system_id, reference, poses[0])
    combined_evaluator = peppr.Evaluator.combine(split_evaluators)
    ref_table = ref_evaluator.tabulate_metrics()
    test_table = combined_evaluator.tabulate_metrics()

    assert test_table.equals(ref_table)


@pytest.mark.parametrize(
    ["model_num", "optimal_bisy_rmsd"],
    [
        (1, 5.82),
        (2, 11.49),
        # For this one model the exhaustive method actually finds the optimal result
        # (3, 7.68),
        (4, 5.38),
        (5, 10.59),
    ],
)
def test_individual_match_method(model_num, optimal_bisy_rmsd):
    """
    Check if :attr:`Evaluator.MatchMethod.INDIVIDUAL` is able to find the optimal metric
    value for a poorly predicted pose, where the other match methods fail.

    The optimal results were computed with OpenStructure.
    """
    TOLERANCE = 0.1

    system_dir = Path(__file__).parent / "data" / "poor" / "7yn2__1__1.A_1.B__1.C"
    pdbx_file = pdbx.CIFFile.read(system_dir / "reference.cif")
    reference = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    pdbx_file = pdbx.CIFFile.read(system_dir / "poses.cif")
    pose = pdbx.get_structure(pdbx_file, model=model_num, include_bonds=True)

    metric = peppr.BiSyRMSD(2.0)
    exhaustive_evaluator = peppr.Evaluator(
        [metric], peppr.Evaluator.MatchMethod.EXHAUSTIVE
    )
    exhaustive_evaluator.feed("foo", reference, pose)
    exhaustive_bisy_rmsd = exhaustive_evaluator.get_results()[0][0]

    individual_evaluator = peppr.Evaluator(
        [metric], peppr.Evaluator.MatchMethod.INDIVIDUAL
    )
    individual_evaluator.feed("foo", reference, pose)
    individual_bisy_rmsd = individual_evaluator.get_results()[0][0]

    # Ensure the validity of this test,
    # i.e. that the heuristic/exhaustive method is really insufficient
    assert exhaustive_bisy_rmsd > optimal_bisy_rmsd + TOLERANCE
    # The individual method should be able to find the optimal value
    assert individual_bisy_rmsd <= optimal_bisy_rmsd + TOLERANCE
