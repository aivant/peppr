__all__ = [
    "Selector",
    "MeanSelector",
    "MedianSelector",
    "OracleSelector",
    "TopSelector",
]

from abc import ABC, abstractmethod
import numpy as np


class Selector(ABC):
    """
    The base class for all pose selectors.

    Its purpose is to aggregate metric values for multiple poses into a single scalar
    value.

    Attributes
    ----------
    name : str
        The name of the selector.
        Used for displaying the results via the :class:`Evaluator`.
        **ABSTRACT:** Must be overridden by subclasses.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def select(self, values: np.ndarray, smaller_is_better: bool) -> float:
        """
        Select the *representative* metric value from a set of poses.

        The meaning of '*representative*' depends on the specific :class:`Selector`
        subclass.

        **ABSTRACT:** Must be overridden by subclasses.

        Parameters
        ----------
        values : ndarray, shape=(n,), dtype=float
            The metric values to select from.
            May contain *NaN* values.
            The values are sorted from highest to lowest confidence.
        smaller_is_better : bool
            Whether the smaller value is considered a better prediction.

        Returns
        -------
        float
            The selected value.

        Notes
        -----
        """
        raise NotImplementedError

    def __hash__(self) -> int:
        return hash(self.name)


class MeanSelector(Selector):
    """
    Selector that computes the mean of the values.
    """

    @property
    def name(self) -> str:
        return "mean"

    def select(self, values: np.ndarray, smaller_is_better: bool) -> float:
        return np.nanmean(values)


class MedianSelector(Selector):
    """
    Selector that computes the median of the values.
    """

    @property
    def name(self) -> str:
        return "median"

    def select(self, values: np.ndarray, smaller_is_better: bool) -> float:
        return np.nanmedian(values)


class OracleSelector(Selector):
    """
    Selector that returns the best value.
    """

    @property
    def name(self) -> str:
        return "Oracle"

    def select(self, values: np.ndarray, smaller_is_better: bool) -> float:
        if smaller_is_better:
            return np.nanmin(values)
        else:
            return np.nanmax(values)


class TopSelector(Selector):
    """
    Selector that returns the best value from the `k` values with highest
    confidence.

    Parameters
    ----------
    k : int
        The best value is chosen from the *k* most confident predictions.
    """

    def __init__(self, k: int) -> None:
        self._k = k

    @property
    def name(self) -> str:
        return f"Top{self._k}"

    def select(self, values: np.ndarray, smaller_is_better: bool) -> float:
        top_values = values[: self._k]
        if smaller_is_better:
            return np.nanmin(top_values)
        else:
            return np.nanmax(top_values)
