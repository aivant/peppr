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


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("smaller_is_better", [False, True])
@pytest.mark.parametrize(
    "selector",
    [
        peppr.MeanSelector(),
        peppr.OracleSelector(),
        peppr.TopSelector(5),
        peppr.TopSelector(1),
        peppr.RandomSelector(5, seed=0),
    ],
    ids=lambda selector: selector.name,
)
def test_nan_values(selector, smaller_is_better):
    """
    Check that :meth:`Selector.select()` returns NaN if all values are NaN, without
    raising a warnings.
    If any value is not NaN, the selector should return an actual value.
    """
    values = np.full(10, np.nan)
    assert np.isnan(selector.select(values, smaller_is_better))
    # Expect a non-NaN value if the input contains any non-NaN value
    values = np.concatenate([np.arange(9), [np.nan]])
    assert not np.isnan(selector.select(values, smaller_is_better))


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
