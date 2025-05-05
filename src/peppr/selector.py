__all__ = [
    "Selector",
    "MeanSelector",
    "MedianSelector",
    "OracleSelector",
    "TopSelector",
    "RandomSelector",
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
        if np.isnan(values).all():
            return np.nan
        return np.nanmean(values)


class MedianSelector(Selector):
    """
    Selector that computes the median of the values.
    """

    @property
    def name(self) -> str:
        return "median"

    def select(self, values: np.ndarray, smaller_is_better: bool) -> float:
        if np.isnan(values).all():
            return np.nan
        return np.nanmedian(values)


class OracleSelector(Selector):
    """
    Selector that returns the best value.
    """

    @property
    def name(self) -> str:
        return "Oracle"

    def select(self, values: np.ndarray, smaller_is_better: bool) -> float:
        if np.isnan(values).all():
            return np.nan
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
        if np.isnan(top_values).all():
            return np.nan
        if smaller_is_better:
            return np.nanmin(top_values)
        else:
            return np.nanmax(top_values)


class RandomSelector(Selector):
    """
    Selector that returns the best value from `k` randomly chosen values.
    Using this selector is equivalent to using the :class:`TopSelector` with
    random confidence values.

    Parameters
    ----------
    k : int
        The best value is chosen from *k* randomly chosen predictions.
    seed : int, optional
        The seed for the random number generator.
        Defaults to 42.
    """

    def __init__(self, k: int, seed: int = 42) -> None:
        self._k = k
        self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return f"Random{self._k}"

    def select(self, values: np.ndarray, smaller_is_better: bool) -> float:
        random_indices = self._rng.choice(
            range(len(values)), size=self._k, replace=False
        )
        top_values = values[random_indices]
        if np.isnan(top_values).all():
            return np.nan
        if smaller_is_better:
            return np.nanmin(top_values)
        else:
            return np.nanmax(top_values)
