__all__ = [
    "SystemSubsetSelector",
    "IsolatedLigandSelector",
]

from abc import ABC, abstractmethod
import biotite.structure as struc
import numpy as np
import numpy.typing as npt


class SystemSubsetSelector(ABC):
    """Class for selecting subsets of a system."""

    @abstractmethod
    def select(self, pose: struc.AtomArray) -> npt.NDArray[np.bool_]:
        """
        Returns masks that can be used to select subsets of the system.

        **ABSTRACT:** Must be overridden by subclasses.

        Parameters
        ----------
        pose : AtomArray, shape=(n,)
            The system to select from.

        Returns
        -------
        NDArray[bool], shape=(n_subsets, n)
            The masks that can be used to select subsets of the system.
            The first dimension of the array is the subset index.
        """
        ...


class IsolatedLigandSelector(SystemSubsetSelector):
    """Selects all isolated ligands in the system."""

    def select(self, pose: struc.AtomArray) -> npt.NDArray[np.bool_]:
        ligand_mask = pose.hetero
        chain_starts = struc.get_chain_starts(pose)
        if len(chain_starts) == 0:
            # No chains in the structure
            return np.array([], dtype=bool)
        chain_masks = struc.get_chain_masks(pose, chain_starts)
        # Only keep chain masks that correspond to ligand chains
        ligand_masks = chain_masks[(chain_masks & ligand_mask).any(axis=-1)]
        return ligand_masks


# TODO: refactor the selectors in metrics.py to be SystemSubsetSelectors
# TODO: refactor metrics to use selectors, agg functions as class attributes
