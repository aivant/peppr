__all__ = [
    "RefFreeMetric",
]

from abc import ABC, abstractmethod
from typing import OrderedDict
import biotite.interface.rdkit as rdkit_interface
import biotite.structure as struc
import numpy as np
from rdkit import Chem
from peppr.sanitize import sanitize
from peppr.system_subset import IsolatedLigandSelector


class RefFreeMetric(ABC):
    """
    The base class for all reference-free evaluation metrics. Extends :class:`Metric`.

    The central :meth:`evaluate()` method takes a pose structure as input
    and returns a scalar score.

    Attributes
    ----------
    name : str
        The name of the metric.
        Used for displaying the results via the :class:`Evaluator`.
        **ABSTRACT:** Must be overridden by subclasses.
    thresholds : dict (str -> float)
        The named thresholds for the metric.
        Each threshold contains the lower bound
    """

    def __init__(self) -> None:
        thresholds = list(self.thresholds.values())
        if sorted(thresholds) != thresholds:
            raise ValueError("Thresholds must be sorted in ascending order")

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    def thresholds(self) -> OrderedDict[str, float]:
        return OrderedDict()

    @abstractmethod
    def evaluate(self, pose: struc.AtomArray) -> float:
        """
        Apply this metric on the given predicted pose.

        **ABSTRACT:** Must be overridden by subclasses.

        Parameters
        ----------
        pose : AtomArray, shape=(n,)
            The predicted pose.

        Returns
        -------
        float
            The metric computed for each pose.
            *NaN*, if the structure is not suitable for this metric.

        Notes
        -----
        Missing atoms in either the reference or the pose can be identified with
        *NaN* values.
        """
        raise NotImplementedError

    @abstractmethod
    def smaller_is_better(self) -> bool:
        """
        Whether as smaller value of this metric is considered a better prediction.

        **ABSTRACT:** Must be overridden by subclasses.

        Returns
        -------
        bool
            If true, a smaller value of this metric is considered a better prediction.
            Otherwise, a larger value is considered a better prediction.
        """
        raise NotImplementedError


class LigandValenceViolations(RefFreeMetric):
    """
    Counts the total number atoms with valance violations in any ligand.

    Uses RDKit's internal valence checks.
    """

    @property
    def name(self) -> str:
        return "Ligand valence violations"

    def smaller_is_better(self) -> bool:
        return True

    def evaluate(self, pose: struc.AtomArray) -> float:
        ligand_selector = IsolatedLigandSelector()
        ligand_masks = ligand_selector.select(pose)
        ligand_atomarrays = [pose[mask] for mask in ligand_masks]
        num_violations_per_ligand = [
            _count_valence_violations_atomarray(aarray) for aarray in ligand_atomarrays
        ]
        return np.sum(num_violations_per_ligand)


def _count_valence_violations(ligand: Chem.Mol) -> int:
    try:
        sanitize(ligand)
    except Exception:
        pass

    return sum(atom.HasValenceViolation() for atom in ligand.GetAtoms())


def _count_valence_violations_atomarray(ligand: struc.AtomArray) -> int:
    mol = rdkit_interface.to_mol(ligand, explicit_hydrogen=False)
    return _count_valence_violations(mol)
