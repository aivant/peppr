__all__ = ["Evaluator", "MatchWarning", "EvaluationWarning"]

import copy
import warnings
from collections.abc import Iterable, Mapping
from typing import Iterator, Sequence
import biotite.structure as struc
import numpy as np
import pandas as pd
from peppr.common import standardize
from peppr.match import find_matching_atoms
from peppr.metric import Metric
from peppr.selector import Selector


class MatchWarning(UserWarning):
    """
    This warning is raised, if a the :class:`Evaluator` fails to match atoms between
    the reference and pose structures.
    """

    pass


class EvaluationWarning(UserWarning):
    """
    This warning is raised, if a :class:`Metric` fails to evaluate a pose.
    """

    pass


class Evaluator(Mapping):
    """
    This class represents the core of :mod:`peppr`.
    Systems are fed via :meth:`feed()` into the :class:`Evaluator`.
    Finally, the evaluation is reported via :meth:`tabulate_metrics()`, which gives a
    scalar metric value for each fed system, or via :meth:`summarize_metrics()`,
    which aggregates the metrics over all systems.

    Parameters
    ----------
    metrics : Iterable of Metric
        The metrics to evaluate the poses against.
        These will make up the columns of the resulting dataframe from
        :meth:`tabulate_metrics()`.
    tolerate_exceptions : bool, optional
        If set to true, exceptions during :class:`Metric.evaluate()` are not propagated.
        Instead a warning is raised and the result is set to ``None``.
    min_sequence_identity : float
        The minimum sequence identity for two chains to be considered the same entity.

    Attributes
    ----------
    metrics : tuple of Metric
        The metrics to evaluate the poses against.
    system_ids : tuple of str
        The IDs of the systems that were fed into the evaluator.
    """

    def __init__(
        self,
        metrics: Iterable[Metric],
        tolerate_exceptions: bool = False,
        min_sequence_identity: float = 0.95,
    ):
        self._metrics = tuple(metrics)
        self._results: list[list[np.ndarray]] = [[] for _ in range(len(metrics))]
        self._ids: list[str] = []
        self._tolerate_exceptions = tolerate_exceptions
        self._min_sequence_identity = min_sequence_identity

    @property
    def metrics(self) -> tuple[Metric, ...]:
        # Use tuple to forbid adding/removing metrics after initialization
        return self._metrics

    @property
    def system_ids(self) -> tuple[str, ...]:
        return tuple(self._ids)

    def feed(
        self,
        system_id: str,
        reference: struc.AtomArray,
        poses: Sequence[struc.AtomArray] | struc.AtomArrayStack | struc.AtomArray,
    ) -> None:
        """
        Evaluate the poses of a system against the reference structure for all metrics.

        Parameters
        ----------
        system_id : str
            The ID of the system that was evaluated.
        reference : AtomArray
            The reference structure of the system.
            Each separate instance/molecule must have a distinct `chain_id`.
        poses : AtomArrayStack or list of AtomArray or AtomArray
            The pose(s) to evaluate.
            It is expected that the poses are sorted from highest to lowest confidence,
            (relevant for :class:`Selector` instances).

        Notes
        -----
        `reference` and `poses` must fulfill the following requirements:

        - The system must have an associated `biotite.structure.BondList`,
          i.e. the ``bonds`` attribute must not be ``None``.
        - Each molecule in the system must have a distinct ``chain_id``.
        - Chains where the ``hetero`` annotation is ``True`` is always interpreted as a
          small molecule.
          Conversely, chains where the ``hetero`` annotation is ``False`` is always
          interpreted as protein or nucleic acid chain.

        The optimal chain mapping and atom mapping in symmetric small molecules
        is handled automatically.
        """
        reference = standardize(reference)
        if isinstance(poses, struc.AtomArray):
            poses = [standardize(poses)]
        elif isinstance(poses, struc.AtomArrayStack):
            poses = list(standardize(poses))
        else:
            poses = [standardize(pose) for pose in poses]
        if len(poses) == 0:
            raise ValueError("No poses provided")

        result_for_system = np.full((len(self._metrics), len(poses)), np.nan)
        for j, pose in enumerate(poses):
            try:
                reference_order, pose_order = find_matching_atoms(
                    reference, pose, min_sequence_identity=self._min_sequence_identity
                )
                matched_reference = reference[reference_order]
                matched_pose = pose[pose_order]
            except Exception as e:
                if self._tolerate_exceptions:
                    warnings.warn(
                        f"Failed to match reference and pose in system '{system_id}': {e}",
                        MatchWarning,
                    )
                    continue
                else:
                    raise
            # Sanity check if something went wrong in the matching
            if not np.array_equal(matched_reference.element, matched_pose.element):
                if self._tolerate_exceptions:
                    warnings.warn(
                        f"'{system_id}' poses could not be matched to reference",
                        EvaluationWarning,
                    )
                else:
                    raise ValueError(
                        f"'{system_id}' poses could not be matched to reference"
                    )

            for i, metric in enumerate(self._metrics):
                try:
                    result_for_system[i, j] = metric.evaluate(
                        matched_reference, matched_pose
                    )
                except Exception as e:
                    if self._tolerate_exceptions:
                        warnings.warn(
                            f"Failed to evaluate {metric.name} on '{system_id}': {e}",
                            EvaluationWarning,
                        )
                    else:
                        raise
        for i, result in enumerate(result_for_system):
            self._results[i].append(result)
        self._ids.append(system_id)

    def get_results(self) -> list[list[np.ndarray]]:
        """
        Return the raw results of the evaluation.

        This includes each metric evaluated on each pose of each system.

        Returns
        -------
        list of list of np.ndarray
            The raw results of the evaluation.
            The outer list iterates over the metrics, the inner list iterates over
            the systems and the array represents the values for each pose.
        """
        return copy.deepcopy(self._results)

    def tabulate_metrics(
        self, selectors: Iterable[Selector] | None = None
    ) -> pd.DataFrame:
        """
        Create a table listing the value for each metric and system.

        Parameters
        ----------
        selectors : list of Selector, optional
            The selectors to use for selecting the best pose of a multi-pose
            prediction.
            This parameter is not necessary if only single-pose predictions were fed
            into the :class:`Evaluator`.

        Returns
        -------
        pandas.DataFrame
            A table listing the value for each metric and system.
            The index is the system ID.

        Examples
        --------

        >>> print(evaluator.tabulate_metrics())
                                           RMSD      lDDT  TM-score
        8ji2__1__1.B__1.J_1.K          2.987937  0.674205  0.883589
        7t4w__1__1.A__1.C             16.762669  0.693087  0.380107
        8jp0__1__1.A__1.B             26.281593  0.510061  0.316204
        7yn2__1__1.A_1.B__1.C          6.657655  0.567117  0.725322
        8oxu__2__1.C__1.E             14.977116  0.339707  0.296535
        7ydq__1__1.A__1.B             26.111820  0.383841  0.360584
        7wuy__1__1.B__1.HA_1.IA_1.OA  16.494774  0.665949  0.666633
        7xh4__1__1.A__1.B_1.C          1.787062  0.748987  0.915388
        7v34__1__1.A__1.C_1.D_1.G      4.472874  0.567491  0.822537
        8jmr__1__1.A_1.B__1.C_1.D      5.058327  0.730324  0.868684
        >>> print(evaluator.tabulate_metrics(OracleSelector()))
                                      CA-RMSD (Oracle)  lDDT (Oracle)  TM-score (Oracle)
        8ji2__1__1.B__1.J_1.K                 2.987937       0.674205           0.883589
        7t4w__1__1.A__1.C                    16.762669       0.693087           0.380107
        8jp0__1__1.A__1.B                    26.281593       0.510061           0.316204
        7yn2__1__1.A_1.B__1.C                 6.657655       0.567117           0.725322
        8oxu__2__1.C__1.E                    14.977116       0.339707           0.296535
        7ydq__1__1.A__1.B                    26.111820       0.383841           0.360584
        7wuy__1__1.B__1.HA_1.IA_1.OA         16.494774       0.665949           0.666633
        7xh4__1__1.A__1.B_1.C                 1.787062       0.748987           0.915388
        7v34__1__1.A__1.C_1.D_1.G             4.472874       0.567491           0.822537
        8jmr__1__1.A_1.B__1.C_1.D             5.058327       0.730324           0.868684
        """
        columns = self._tabulate_metrics(selectors)
        # Convert (metric, selector)-tuples to strings
        columns = {
            (
                f"{metric.name} ({selector.name})"
                if selector is not None
                else metric.name
            ): values
            for (metric, selector), values in columns.items()
        }
        return pd.DataFrame(columns, index=self._ids)

    def summarize_metrics(
        self, selectors: Iterable[Selector] | None = None
    ) -> dict[str, float]:
        """
        Condense the system-wise evaluation to scalar values for each metric.

        For each metric,

        - the mean value
        - the median value
        - and the percentage of systems within each threshold

        is computed.

        Parameters
        ----------
        selectors : list of Selector, optional
            The selectors to use for selecting the best pose of a multi-pose
            prediction.
            This parameter is not necessary if only single-pose predictions were fed
            into the :class:`Evaluator`.

        Returns
        -------
        dict (str -> float)
            A dictionary mapping the summarized metric name to the scalar value.
            The summarized metric name contains

            - the metric name (e.g. ``DockQ``)
            - the selector name, if a selector was used (e.g. ``Oracle``)
            - the threshold (if a threshold was used) (e.g. ``% acceptable``)

        Examples
        --------

        >>> import pprint
        >>> pprint.pprint(evaluator.summarize_metrics())
        {'CA-RMSD <5.0': 0.3,
         'CA-RMSD >5.0': 0.7,
         'CA-RMSD mean': 12.159182685504375,
         'TM-score mean': 0.6235582438144873,
         'lDDT mean': 0.5880769924413414}
        >>> pprint.pprint(evaluator.summarize_metrics([MeanSelector(), OracleSelector()]))
        {'CA-RMSD <5.0 (Oracle)': 0.3,
         'CA-RMSD <5.0 (mean)': 0.3,
         'CA-RMSD >5.0 (Oracle)': 0.7,
         'CA-RMSD >5.0 (mean)': 0.7,
         'CA-RMSD mean (Oracle)': 12.159182685504375,
         'CA-RMSD mean (mean)': 12.159182685504375,
         'TM-score mean (Oracle)': 0.6235582438144873,
         'TM-score mean (mean)': 0.6235582438144873,
         'lDDT mean (Oracle)': 0.5880769924413414,
         'lDDT mean (mean)': 0.5880769924413414}
        """
        columns = self._tabulate_metrics(selectors)
        output_columns = {}
        for (metric, selector), values in columns.items():
            if metric.thresholds:
                edges = list(metric.thresholds.values()) + [np.inf]
                counts_per_bin, _ = np.histogram(values, bins=edges)
                # NaN values do not bias the percentages,
                # as they are not included in any bin
                percentages_per_bin = counts_per_bin / np.sum(counts_per_bin)
                for threshold_name, percentage in zip(
                    metric.thresholds.keys(), percentages_per_bin
                ):
                    column_name = f"{metric.name} {threshold_name}"
                    if selector is not None:
                        column_name += f" ({selector.name})"
                    output_columns[column_name] = percentage.item()
            # Always add the mean and median value as well
            for name, function in [("mean", np.nanmean), ("median", np.nanmedian)]:
                column_name = f"{metric.name} {name}"
                if selector is not None:
                    column_name += f" ({selector.name})"
                output_columns[column_name] = function(values).item()  # type: ignore[operator]
        return output_columns

    def _tabulate_metrics(
        self, selectors: Iterable[Selector] | None = None
    ) -> dict[tuple[Metric, Selector | None], np.ndarray]:
        columns: dict[tuple[Metric, Selector | None], np.ndarray] = {}
        for i, metric in enumerate(self._metrics):
            values = self._results[i]
            if not selectors:
                condensed_values = []
                for array in values:
                    if array is None:
                        condensed_values.append(np.nan)
                    elif len(array) > 1:
                        raise ValueError(
                            "At least one selector is required for multi-pose predictions"
                        )
                    else:
                        condensed_values.append(array[0])
                columns[metric, None] = np.array(condensed_values)
            else:
                for selector in selectors:
                    condensed_values = np.array(
                        [
                            selector.select(val, metric.smaller_is_better())
                            if val is not None
                            else np.nan
                            for val in values
                        ]
                    )  # type: ignore[assignment]
                    columns[metric, selector] = condensed_values  # type: ignore[assignment]
        return columns

    def __getitem__(self, metric_name: str) -> list[np.ndarray]:
        return self._results[self._metrics.index(metric_name)]

    def __iter__(self) -> Iterator[list[np.ndarray]]:
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)
