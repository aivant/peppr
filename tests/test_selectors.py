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
    Check that the :class:`RandomSelector`, when ran multiple times, selects approximately the expected value for known
    examples.
    """
    selector = peppr.Selector.RandomSelector(k=5)
    values = np.linspace(0, 10, 10 + 1)

    selected_values = [
        selector.select(values, smaller_is_better=False) for _ in range(20)
    ]

    assert np.isclose(np.mean(selected_values), 5.0, rtol=0.5)
