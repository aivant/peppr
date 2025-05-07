import numpy as np
import pytest
import peppr


@pytest.mark.parametrize(
    ["selector", "expected_value"],
    [
        (peppr.MeanSelector(), 5.0),
        (
            peppr.OracleSelector(),
            10,
        ),
        (peppr.TopSelector(5), 4),
        (peppr.TopSelector(1), 0),
        (peppr.RandomSelector(5, seed=0), 7),
    ],
    ids=lambda x: x.name if isinstance(x, peppr.Selector) else "",
)
def test_selectors(selector, expected_value):
    """
    Check for each implemented selector whether :meth:`Selector.select()` selects
    the expected value for known examples.
    """
    values = np.linspace(0, 10, 10 + 1)
    selected_value = selector.select(values, smaller_is_better=False)

    assert selected_value == expected_value


def test_random_selector():
    """
    Test the RandomSelector's statistical behavior.

    This test verifies that when a RandomSelector with k=5 is used multiple times
    on the same data, the average of selected values approximates the expected value
    of the maximum of 5 samples drawn without replacement from a uniform distribution
    from 0 to 10, which is 9.
    The test allows for some variance with a relative tolerance of 0.5.
    """
    selector = peppr.RandomSelector(k=5)
    values = np.linspace(0, 10, 10 + 1)

    selected_values = [
        selector.select(values, smaller_is_better=False) for _ in range(20)
    ]

    assert np.isclose(np.mean(selected_values), 9, rtol=0.5)


def test_variance_selector():
    """
    This test verifies that the VarianceSelector returns the expected value of
    variance for a given set of values.
    """
    selector = peppr.VarianceSelector()
    values = np.linspace(0, 10, 10 + 1)
    expected_variance = np.var(values)

    selected_value = selector.select(values, smaller_is_better=False)

    assert np.isclose(selected_value, expected_variance)
