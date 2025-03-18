import copy
import itertools
from collections import OrderedDict
import numpy as np
import pytest
import peppr
from tests.common import assemble_predictions, list_test_predictions


@pytest.fixture
def metrics():
    return [peppr.MonomerRMSD(5.0), peppr.MonomerLDDTScore(), peppr.MonomerTMScore()]


@pytest.fixture
def selectors():
    return [peppr.MeanSelector(), peppr.OracleSelector()]


@pytest.fixture
def evaluator(metrics):
    return peppr.Evaluator(metrics)


@pytest.fixture
def evaluator_fed_with_single_models(evaluator):
    evaluator = copy.deepcopy(evaluator)
    system_ids = list_test_predictions()
    for system_id in system_ids:
        reference, models = assemble_predictions(system_id)
        evaluator.feed(system_id, reference, models[0])
    return evaluator


@pytest.fixture
def evaluator_fed_with_multiple_models(evaluator):
    evaluator = copy.deepcopy(evaluator)
    system_ids = list_test_predictions()
    for system_id in system_ids:
        reference, models = assemble_predictions(system_id)
        evaluator.feed(system_id, reference, models)
    return evaluator


def test_tabulate_metrics_for_single(evaluator_fed_with_single_models):
    """
    Check if :meth:`Evaluator.tabulate_metrics()` returns a dataframe with the correct
    system IDs and column names, if a single model is fed for each system.
    """
    table = evaluator_fed_with_single_models.tabulate_metrics()

    assert table.index.to_list() == list_test_predictions()
    assert table.columns.to_list() == ["CA-RMSD", "intra protein lDDT", "TM-score"]


def test_tabulate_metrics_for_multi(evaluator_fed_with_multiple_models, selectors):
    """
    Check if :meth:`Evaluator.tabulate_metrics()` returns a dataframe with the correct
    system IDs and column names, if multiple models are fed for each system.
    """
    table = evaluator_fed_with_multiple_models.tabulate_metrics(selectors)

    assert table.index.to_list() == list_test_predictions()
    assert table.columns.to_list() == [
        "CA-RMSD (mean)",
        "CA-RMSD (Oracle)",
        "intra protein lDDT (mean)",
        "intra protein lDDT (Oracle)",
        "TM-score (mean)",
        "TM-score (Oracle)",
    ]


@pytest.mark.parametrize("selectors", [None, [peppr.MeanSelector()]])
def test_tabulate_metrics_with_unsuitable_metric(selectors):
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

        def evaluate(self, reference, models):
            return None

        def requires_hydrogen(self):
            return False

        def smaller_is_better(self):
            return False

    evaluator = peppr.Evaluator([peppr.MonomerRMSD(5.0), UnsuitableMetric()])
    system_ids = list_test_predictions()
    for system_id in system_ids:
        reference, models = assemble_predictions(system_id)
        evaluator.feed(system_id, reference, models[0])
    table = evaluator.tabulate_metrics(selectors)

    # The first metric (MonomerRMSD) should work
    assert table.iloc[:, 0].notna().all()
    # But the second metric (UnsuitableMetric) should not
    assert table.iloc[:, 1].isna().all()


@pytest.mark.parametrize("single_model", [True, False])
def test_summarize_metrics(
    single_model,
    evaluator_fed_with_single_models,
    evaluator_fed_with_multiple_models,
    selectors,
):
    """
    Check if :meth:`Evaluator.summarize_metrics()` returns a dictionary that maps
    the expected metric names to floating point values, when fed with a single model
    per system.
    """
    METRIC_NAMES = [
        "CA-RMSD <5.0",
        "CA-RMSD >5.0",
        "CA-RMSD mean",
        "CA-RMSD median",
        "intra protein lDDT mean",
        "intra protein lDDT median",
        "TM-score mean",
        "TM-score median",
    ]
    SELECTOR_NAMES = ["mean", "Oracle"]

    if single_model:
        summary = evaluator_fed_with_single_models.summarize_metrics()
    else:
        summary = evaluator_fed_with_multiple_models.summarize_metrics(selectors)

    if single_model:
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


def test_summarize_metrics_with_nan():
    """
    Check if :meth:`Evaluator.summarize_metrics()` is able to handle metrics that
    return `None` for some systems.
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

        def evaluate(self, reference, models):
            if self._works:
                self._works = False
                return self._rng.random()
            else:
                self._works = True
                return None

        def requires_hydrogen(self):
            return False

        def smaller_is_better(self):
            return False

    evaluator = peppr.Evaluator([SometimesUnsuitableMetric()])
    system_ids = list_test_predictions()
    for system_id in system_ids:
        reference, models = assemble_predictions(system_id)
        evaluator.feed(system_id, reference, models[0])
    summary = evaluator.summarize_metrics()

    # Although the metric is NaN for half of the systems,
    # the summary should still contain finite values, ignoring NaNs
    assert np.isfinite(list(summary.values())).all()
    # Summing the bins should give 1.0, i.e. NaNs are ignored in computing percentages
    assert summary["Metric bad"] + summary["Metric good"] == pytest.approx(1.0)


def test_tolerate_exceptions():
    """
    Check if ``tolerate_exceptions`` leads to warnings instead of exceptions.
    """

    class BrokenMetric(peppr.Metric):
        """
        Placeholder metric that always fails.
        """

        @property
        def name(self):
            return "Broken"

        def evaluate(self, reference, models):
            raise ValueError("Expected failure")

        def requires_hydrogen(self):
            return False

        def smaller_is_better(self):
            return False

    evaluator = peppr.Evaluator([BrokenMetric()], tolerate_exceptions=True)
    system_id = list_test_predictions()[0]
    reference, models = assemble_predictions(system_id)
    with pytest.warns(
        peppr.EvaluationWarning,
        match=f"Failed to evaluate Broken on '{system_id}': Expected failure",
    ):
        evaluator.feed(system_id, reference, models[0])
