__all__ = ["Evaluator", "MatchWarning", "EvaluationWarning"]

import copy
import itertools
import warnings
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Iterator, Sequence
import biotite.structure as struc
import numpy as np
import pandas as pd
from peppr.common import standardize
from peppr.match import find_all_matches, find_optimal_match
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
    match_method : MatchMethod, optional
        The strategy to use for finding atom matches between the reference and pose.
        This can be used to trade off speed and accuracy.
    max_matches : int, optional
        The maximum number of atom matches to try, if the `match_method` is set to
        ``EXHAUSTIVE`` or ``INDIVIDUAL``.
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

    class MatchMethod(Enum):
        """
        Method for finding atom matches between the fed reference and pose.
        These methods represent a tradeoff between speed and accuracy.

        - ``HEURISTIC``: Use a fast heuristic [1]_ that matches the reference and pose
          by minimizing the RMSD between the centroids of each chain.
          This method is fast and scales linearly with the number of chains, but
          it is not guaranteed to find the optimal match in all cases, especially
          when the pose and reference are quite distant from each other.
        - ``EXHAUSTIVE``: Exhaustively iterate through all valid atom mappings
          between the reference and pose and select the one that gives the lowest
          all-atom RMSD.
          This method is slower and prone to combinatorial explosion, but it finds
          better matches in edge cases.
        - ``INDIVIDUAL``: Like ``EXHAUSTIVE``, but instead of using the RMSD as
          criterion for optimization, each individual :class:`Metric` is used.
          As this requires exhaustive iteration over all mappings and computing the
          each metric for all of them, this method is slower than ``EXHAUSTIVE``.
          However, it guarantees to find the optimal match for each metric.
        - ``NONE``: Skip atom matching entirely and evaluate metrics on the
          structures as provided. This is useful when the reference and pose are
          already properly aligned, or when using metrics that don't require
          matching (e.g., bond-length violations, clash counts).

        References
        ----------
        .. [1] *Protein complex prediction with AlphaFold-Multimer*, Section 7.3, https://doi.org/10.1101/2021.10.04.463034
        """

        HEURISTIC = "heuristic"
        EXHAUSTIVE = "exhaustive"
        INDIVIDUAL = "individual"
        NONE = "none"

    def __init__(
        self,
        metrics: Iterable[Metric],
        match_method: MatchMethod = MatchMethod.HEURISTIC,
        max_matches: int | None = None,
        tolerate_exceptions: bool = False,
        min_sequence_identity: float = 0.95,
    ):
        self._metrics = tuple(metrics)
        self._match_method = match_method
        self._max_matches = max_matches
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

    @staticmethod
    def combine(evaluators: Iterable["Evaluator"]) -> "Evaluator":
        """
        Combine multiple :class:`Evaluator` instances into a single one,
        preserving the systems fed to each instance.

        Parameters
        ----------
        evaluators : Iterable of Evaluator
            The evaluators to combine.
            The ``metrics``, ``tolerate_exceptions`` and ``min_sequence_identity``
            must be the same for all evaluators.

        Returns
        -------
        Evaluator
            The evaluator combining the systems of all input `evaluators` in the order
            of the input.
        """
        ref_evaluator = None
        all_ids = []
        all_results = []
        for evaluator in evaluators:
            if ref_evaluator is None:
                ref_evaluator = evaluator
            if evaluator != ref_evaluator:
                raise ValueError(
                    "All evaluators must be initialized with the same parameters"
                )
            all_ids.append(evaluator.system_ids)
            all_results.append(evaluator.get_results())
        if ref_evaluator is None:
            raise ValueError("At least one evaluator must be provided")

        combined_evaluator = copy.deepcopy(ref_evaluator)
        combined_evaluator._ids = list(itertools.chain(*all_ids))
        combined_evaluator._results = [
            list(itertools.chain(*[results[i] for results in all_results]))
            for i in range(len(combined_evaluator._metrics))
        ]
        return combined_evaluator

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

        The optimal atom matching is handled automatically based on the
        :class:`MatchMethod`.
        """
        try:
            reference = standardize(reference)
            if isinstance(poses, struc.AtomArray):
                poses = [standardize(poses)]
            elif isinstance(poses, struc.AtomArrayStack):
                poses = list(standardize(poses))
            else:
                poses = [standardize(pose) for pose in poses]
            if len(poses) == 0:
                raise ValueError("No poses provided")
        except Exception as e:
            self._raise_or_warn(
                e,
                UserWarning(f"Failed to standardize system '{system_id}': {e}"),
            )

        if self._match_method in (
            Evaluator.MatchMethod.HEURISTIC,
            Evaluator.MatchMethod.EXHAUSTIVE,
        ):
            use_heuristic = self._match_method == Evaluator.MatchMethod.HEURISTIC
            result_for_system = np.stack(
                [
                    self._evaluate_using_rmsd(system_id, reference, pose, use_heuristic)
                    for pose in poses
                ],
                axis=-1,
            )
        elif self._match_method == Evaluator.MatchMethod.INDIVIDUAL:
            result_for_system = np.stack(
                [
                    self._evaluate_using_each_metric(system_id, reference, pose)
                    for pose in poses
                ],
                axis=-1,
            )
        elif self._match_method == Evaluator.MatchMethod.NONE:
            result_for_system = np.stack(
                [
                    self._evaluate_without_matching(system_id, reference, pose)
                    for pose in poses
                ],
                axis=-1,
            )
        else:
            raise ValueError(f"Unknown match strategy: {self._match_method}")

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

    def __getitem__(self, metric_name: str) -> list[np.ndarray]:
        return self._results[self._metrics.index(metric_name)]

    def __iter__(self) -> Iterator[list[np.ndarray]]:
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Evaluator):
            return False
        if len(self._metrics) != len(other._metrics):
            return False
        for metric, ref_metric in zip(self._metrics, other._metrics):
            if metric.name != ref_metric.name:
                return False
        if self._tolerate_exceptions != other._tolerate_exceptions:
            return False
        if self._min_sequence_identity != other._min_sequence_identity:
            return False
        return True

    def _evaluate_using_rmsd(
        self,
        system_id: str,
        reference: struc.AtomArray,
        pose: struc.AtomArray,
        use_heuristic: bool,
    ) -> np.ndarray:
        """
        Run the metrics on the given system using matching that aims to minimize the
        RMSD between the reference and pose.
        """
        results = np.full(len(self._metrics), np.nan)
        try:
            reference_order, pose_order = find_optimal_match(
                reference,
                pose,
                min_sequence_identity=self._min_sequence_identity,
                use_heuristic=use_heuristic,
                max_matches=self._max_matches,
            )
            matched_reference = reference[reference_order]
            matched_pose = pose[pose_order]
            # Sanity check if something went wrong in the matching
            if not np.array_equal(matched_reference.element, matched_pose.element):
                raise ValueError("Reference and pose arrays are not array_equal")
        except Exception as e:
            self._raise_or_warn(
                e,
                MatchWarning(
                    f"Failed to match reference and pose in system '{system_id}': {e}"
                ),
            )

        for i, metric in enumerate(self._metrics):
            try:
                results[i] = metric.evaluate(matched_reference, matched_pose)
            except Exception as e:
                self._raise_or_warn(
                    e,
                    EvaluationWarning(
                        f"Failed to evaluate {metric.name} on '{system_id}': {e}"
                    ),
                )
        return results

    def _evaluate_without_matching(
        self, system_id: str, reference: struc.AtomArray, pose: struc.AtomArray
    ) -> np.ndarray:
        """
        Run the metrics on the given system without performing any atom matching.

        This assumes that the reference and pose structures are already properly
        aligned with matching atom ordering, or that the metrics don't require matching.
        """
        results = np.full(len(self._metrics), np.nan)

        for i, metric in enumerate(self._metrics):
            try:
                results[i] = metric.evaluate(reference, pose)
            except Exception as e:
                self._raise_or_warn(
                    e,
                    EvaluationWarning(
                        f"Failed to evaluate {metric.name} on '{system_id}': {e}"
                    ),
                )
        return results

    def _evaluate_using_each_metric(
        self, system_id: str, reference: struc.AtomArray, pose: struc.AtomArray
    ) -> np.ndarray:
        """
        Run the metrics on the given system using a separate matching for each metric,
        that specifically optimizes the given metric.
        """

        class _FindAllMatchesError(Exception):
            """
            An exception class to be able to specifically catch exceptions in the
            loop head calling `find_all_matches()`.

            Parameters
            ----------
            wrapped_exception : Exception
                The actual exception raised by `find_all_matches()`.
            """

            def __init__(self, wrapped_exception):  # type: ignore[no-untyped-def]
                self.wrapped_exception = wrapped_exception

        def _find_all_matches(*args, **kwargs):  # type: ignore[no-untyped-def]
            try:
                for e in find_all_matches(*args, **kwargs):
                    yield e
            except Exception as e:
                raise _FindAllMatchesError(e)

        results = np.full(len(self._metrics), np.nan)

        for i, metric in enumerate(self._metrics):
            try:
                best_result = np.inf if metric.smaller_is_better() else -np.inf
                for it, (reference_order, pose_order) in enumerate(
                    _find_all_matches(
                        reference,
                        pose,
                        min_sequence_identity=self._min_sequence_identity,
                    )
                ):
                    if self._max_matches is not None and it >= self._max_matches:
                        break
                    matched_reference = reference[reference_order]
                    matched_pose = pose[pose_order]
                    # Sanity check if something went wrong in the matching
                    if not np.array_equal(
                        matched_reference.element, matched_pose.element
                    ):
                        self._raise_or_warn(
                            ValueError(
                                f"'{system_id}' poses could not be matched to reference"
                            ),
                            MatchWarning,
                        )
                        best_result = np.nan
                        break
                    try:
                        result = metric.evaluate(matched_reference, matched_pose)
                    except Exception as e:
                        self._raise_or_warn(
                            e,
                            EvaluationWarning(
                                f"Failed to evaluate {metric.name} on '{system_id}': "
                                f"{e}"
                            ),
                        )
                        best_result = np.nan
                        break
                    if metric.smaller_is_better():
                        if result < best_result:
                            best_result = result
                    else:
                        if result > best_result:
                            best_result = result

            except _FindAllMatchesError as e:
                self._raise_or_warn(
                    e.wrapped_exception,
                    MatchWarning(
                        f"Failed to match reference and pose in system '{system_id}': "
                        f"{e}"
                    ),
                )
                best_result = np.nan

            if np.isfinite(best_result):
                results[i] = best_result

        return results

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

    def _raise_or_warn(
        self, exception: Exception, alternative_warning: Warning | type[Warning]
    ) -> None:
        """
        Raise the given exception, if ``tolerate_exceptions`` is set to ``False``,
        or raise a warning otherwise.

        Parameters
        ----------
        exception : Exception
            The exception to raise.
        alternative_warning : Warning or type[Warning]
            The warning to raise if ``tolerate_exceptions`` is set to ``False``.
            If only :class:`Warning` type is given instead of an instance, the warning
            message is taken from the `exception`.
        """
        if self._tolerate_exceptions:
            if isinstance(alternative_warning, type):
                warnings.warn(str(exception), alternative_warning)
            else:
                warnings.warn(alternative_warning)
        else:
            raise exception
