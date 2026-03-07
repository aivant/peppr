__all__ = [
    "Metric",
    "MonomerRMSD",
    "MonomerTMScore",
    "MonomerLDDTScore",
    "IntraLigandLDDTScore",
    "LDDTPLIScore",
    "LDDTPPIScore",
    "GlobalLDDTScore",
    "DockQScore",
    "LigandRMSD",
    "InterfaceRMSD",
    "ContactFraction",
    "PocketAlignedLigandRMSD",
    "BiSyRMSD",
    "BondLengthViolations",
    "BondAngleViolations",
    "ClashCount",
    "PLIFRecovery",
    "ChiralityViolations",
    "PocketDistance",
    "PocketVolumeOverlap",
    "RotamerViolations",
    "RamachandranViolations",
    "LigandValenceViolations",
]

import itertools
import warnings
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from enum import Enum, auto
from typing import Any, Callable, Dict
import biotite.interface.rdkit as rdkit_interface
import biotite.structure as struc
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from peppr.bisyrmsd import bisy_rmsd
from peppr.bounds import get_distance_bounds
from peppr.clashes import find_clashes
from peppr.common import (
    ACCEPTOR_PATTERN,
    DONOR_PATTERN,
    HALOGEN_DISTANCE_SCALING,
    HALOGEN_PATTERN,
    HBOND_DISTANCE_SCALING,
    is_small_molecule,
)
from peppr.contacts import ContactMeasurement
from peppr.dockq import (
    dockq,
    fnat,
    get_contact_residues,
    irmsd,
    lrmsd,
    pocket_aligned_lrmsd,
)
from peppr.match import filter_matched, find_matching_centroids
from peppr.rotamer import (
    get_fraction_of_rama_outliers,
    get_fraction_of_rotamer_outliers,
)
from peppr.sanitize import sanitize
from peppr.volume import volume_overlap


class Metric(ABC):
    """
    The base class for all evaluation metrics.

    The central :meth:`evaluate()` method takes a for a system reference and pose
    structures as input and returns a sclar score.

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
    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        """
        Apply this metric on the given predicted pose with respect to the given
        reference.

        **ABSTRACT:** Must be overridden by subclasses.

        Parameters
        ----------
        reference : AtomArray, shape=(n,)
            The reference structure of the system.
            Each separate instance/molecule must have a distinct `chain_id`.
        pose : AtomArray, shape=(n,)
            The predicted pose.
            Must have the same length and atom order as the `reference`.

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

    def disable_atom_matching(self) -> bool:
        """
        Defines whether the upstream atom matching is disabled for this metric.

        DEPRECATED: This method has no effect anymore.

        Returns
        -------
        bool
            No effect.
        """
        warnings.warn("The method has no effect anymore", DeprecationWarning)
        return False


class MonomerRMSD(Metric):
    r"""
    Compute the *root mean squared deviation* (RMSD) between each peptide chain in the
    reference and the pose and take the mean weighted by the number of heavy atoms.

    Parameters
    ----------
    threshold : float
        The RMSD threshold to use for the *good* predictions.
    backbone_only : bool, optional
        If ``True``, only consider :math:`C_{\alpha}` from peptides and :math:`C_3^'`
        from nucleic acids.
        Otherwise, consider all heavy atoms.
    """

    def __init__(self, threshold: float, backbone_only: bool = True) -> None:
        self._threshold = threshold
        self._backbone_only = backbone_only
        super().__init__()

    @property
    def name(self) -> str:
        if self._backbone_only:
            return "backbone RMSD"
        else:
            return "all-atom RMSD"

    @property
    def thresholds(self) -> OrderedDict[str, float]:
        return OrderedDict(
            [(f"<{self._threshold}", 0), (f">{self._threshold}", self._threshold)]
        )

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        def superimpose_and_rmsd(reference_chain, pose_chain):  # type: ignore[no-untyped-def]
            pose_chain, _ = struc.superimpose(reference_chain, pose_chain)
            return struc.rmsd(reference_chain, pose_chain)

        if self._backbone_only:
            reference, pose = filter_matched(
                reference,
                pose,
                prefilter=lambda atoms: ~atoms.hetero
                & np.isin(atoms.atom_name, ["CA", "C3'"]),
            )
        else:
            reference, pose = filter_matched(
                reference, pose, prefilter=lambda atoms: ~atoms.hetero
            )
        return _run_for_each_monomer(reference, pose, superimpose_and_rmsd)

    def smaller_is_better(self) -> bool:
        return True


class MonomerTMScore(Metric):
    """
    Compute the *TM-score* score for each protein monomer and take the mean weighted
    by the number of atoms.
    """

    @property
    def name(self) -> str:
        return "TM-score"

    @property
    def thresholds(self) -> OrderedDict[str, float]:
        # Bins are adopted from https://doi.org/10.1093/nar/gki524
        return OrderedDict(
            [
                ("random", 0.00),
                ("ambiguous", 0.17),
                ("similar", 0.50),
            ]
        )

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        def superimpose_and_tm_score(reference, pose):  # type: ignore[no-untyped-def]
            # Use 'superimpose_structural_homologs()' instead of 'superimpose()',
            # as it optimizes the TM-score instead of the RMSD
            try:
                super, _, ref_i, pose_i = struc.superimpose_structural_homologs(
                    reference, pose, max_iterations=1
                )
                return struc.tm_score(reference, super, ref_i, pose_i)
            except ValueError as e:
                if "No anchors found" in str(e):
                    # The structures are too dissimilar for structure-based
                    # superimposition, i.e. the pose is very bad
                    return 0.0
                else:
                    raise

        reference, pose = filter_matched(
            reference,
            pose,
            # TM-score is only defined for peptide chains
            prefilter=lambda atoms: struc.filter_amino_acids(atoms) & ~atoms.hetero,
        )
        return _run_for_each_monomer(reference, pose, superimpose_and_tm_score)

    def smaller_is_better(self) -> bool:
        return False


class MonomerLDDTScore(Metric):
    """
    Compute the *local Distance Difference Test* (lDDT) score for each monomer
    and take the mean weighted by the number of atoms.
    """

    @property
    def name(self) -> str:
        return "intra polymer lDDT"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        reference, pose = filter_matched(
            reference, pose, prefilter=lambda atoms: ~atoms.hetero
        )
        return _run_for_each_monomer(reference, pose, struc.lddt)

    def smaller_is_better(self) -> bool:
        return False


class IntraLigandLDDTScore(Metric):
    """
    Compute the *local Distance Difference Test* (lDDT) score for contacts within each
    small molecule.
    """

    @property
    def name(self) -> str:
        return "intra ligand lDDT"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        def within_same_molecule(contacts: np.ndarray) -> np.ndarray:
            # Find the index of the chain/molecule for each atom
            chain_indices = struc.get_chain_positions(
                reference, contacts.flatten()
            ).reshape(contacts.shape)
            # Remove contacts between atoms of different molecules
            return chain_indices[:, 0] == chain_indices[:, 1]

        reference, pose = filter_matched(
            reference, pose, prefilter=lambda atoms: atoms.hetero
        )
        if reference.array_length() == 0:
            # No ligands present
            return np.nan
        return struc.lddt(
            reference,
            pose,
            exclude_same_residue=False,
            filter_function=within_same_molecule,
        ).item()

    def smaller_is_better(self) -> bool:
        return False


class LDDTPLIScore(Metric):
    """
    Compute the CASP LDDT-PLI score, i.e. the lDDT for polymer-ligand interactions
    as defined by [1]_.

    References
    ----------
    .. [1] https://doi.org/10.1002/prot.26601
    """

    @property
    def name(self) -> str:
        return "polymer-ligand lDDT"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        def lddt_pli(reference, pose):  # type: ignore[no-untyped-def]
            ligand_mask = reference.hetero
            polymer_mask = ~ligand_mask

            if not polymer_mask.any():
                # No polymer present -> metric is undefined
                return np.nan

            binding_site_contacts = np.unique(
                get_contact_residues(
                    reference[polymer_mask], reference[ligand_mask], cutoff=4.0
                )[:, 0]
            )
            # No contacts between the ligand and polymer in reference -> no binding site
            # -> metric is undefined
            if len(binding_site_contacts) == 0:
                return np.nan
            binding_site_mask = struc.get_residue_masks(
                reference, binding_site_contacts
            ).any(axis=0)
            return struc.lddt(
                reference,
                pose,
                atom_mask=ligand_mask,
                partner_mask=binding_site_mask,
                inclusion_radius=6.0,
                distance_bins=(0.5, 1.0, 2.0, 4.0),
                symmetric=True,
            )

        reference, pose = filter_matched(reference, pose)
        return _average_over_ligands(reference, pose, lddt_pli)

    def smaller_is_better(self) -> bool:
        return False


class LDDTPPIScore(Metric):
    """
    Compute the the lDDT for polymer-polymer interactions, i.e. all intra-chain
    contacts are not included.
    """

    @property
    def name(self) -> str:
        return "polymer-polymer lDDT"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        reference, pose = filter_matched(
            reference, pose, prefilter=lambda atoms: ~atoms.hetero
        )
        if reference.array_length() == 0:
            # This is not a PPI system
            return np.nan
        return struc.lddt(reference, pose, exclude_same_chain=True)

    def smaller_is_better(self) -> bool:
        return False


class GlobalLDDTScore(Metric):
    r"""
    Compute the lDDT score for all contacts in the system, i.e. both intra- and
    inter-chain contacts.
    This is equivalent to the original lDDT definition in [1]_.

    Parameters
    ----------
    backbone_only : bool, optional
        If ``True``, only consider :math:`C_{\alpha}` from peptides and :math:`C_3^'`
        from nucleic acids.
        Otherwise, consider all heavy atoms.

    References
    ----------
    .. [1] https://doi.org/10.1093/bioinformatics/btt473
    """

    def __init__(self, backbone_only: bool = True) -> None:
        self._backbone_only = backbone_only
        super().__init__()

    @property
    def name(self) -> str:
        if self._backbone_only:
            return "global backbone lDDT"
        else:
            return "global all-atom lDDT"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        if self._backbone_only:
            reference, pose = filter_matched(
                reference,
                pose,
                prefilter=lambda atoms: ~atoms.hetero
                & np.isin(atoms.atom_name, ["CA", "C3'"]),
            )
        else:
            reference, pose = filter_matched(reference, pose)
        if reference.array_length() == 0:
            return np.nan
        return struc.lddt(reference, pose).item()

    def smaller_is_better(self) -> bool:
        return False


class DockQScore(Metric):
    """
    Compute the *DockQ* score for the given complex as defined in [1]_.

    Parameters
    ----------
    include_pli : bool, optional
        If set to ``False``, small molecules are excluded from the calculation.

    References
    ----------
    .. [1] https://doi.org/10.1093/bioinformatics/btae586
    """

    def __init__(self, include_pli: bool = True) -> None:
        self._include_pli = include_pli
        super().__init__()

    @property
    def name(self) -> str:
        if self._include_pli:
            return "DockQ"
        else:
            return "DockQ-PPI"

    @property
    def thresholds(self) -> OrderedDict[str, float]:
        return OrderedDict(
            [
                ("incorrect", 0.0),
                ("acceptable", 0.23),
                ("medium", 0.49),
                ("high", 0.80),
            ]
        )

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        def run_dockq(reference_chain1, reference_chain2, pose_chain1, pose_chain2):  # type: ignore[no-untyped-def]
            if not self._include_pli and (
                is_small_molecule(reference_chain1)
                or is_small_molecule(reference_chain2)
            ):
                # Do not compute DockQ for PLI pairs if disabled
                return np.nan
            if is_small_molecule(reference_chain1) and is_small_molecule(
                reference_chain2
            ):
                # Do not compute DockQ for small molecule pairs
                return np.nan
            return dockq(
                *_select_receptor_and_ligand(
                    reference_chain1, reference_chain2, pose_chain1, pose_chain2
                )
            ).score

        reference, pose = filter_matched(reference, pose)
        return _run_for_each_chain_pair(reference, pose, run_dockq)

    def smaller_is_better(self) -> bool:
        return False


class LigandRMSD(Metric):
    """
    Compute the *Ligand RMSD* for the given polymer complex as defined in [1]_.
    The score is first separately computed for all pairs of chains that are in contact,
    and the averaged. If the reference doesn't contain any chains in contact, *NaN* is returned.

    References
    ----------
    .. [1] https://doi.org/10.1093/bioinformatics/btae586
    """

    @property
    def name(self) -> str:
        return "LRMSD"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        def lrmsd_on_interfaces_only(
            reference_chain1: struc.AtomArray,
            reference_chain2: struc.AtomArray,
            pose_chain1: struc.AtomArray,
            pose_chain2: struc.AtomArray,
        ) -> float | np.floating | NDArray[np.floating]:
            reference_contacts = get_contact_residues(
                reference_chain1,
                reference_chain2,
                cutoff=10.0,
            )

            if len(reference_contacts) == 0:
                return np.nan
            else:
                return lrmsd(
                    *_select_receptor_and_ligand(
                        reference_chain1, reference_chain2, pose_chain1, pose_chain2
                    )
                )

        reference, pose = filter_matched(
            reference, pose, prefilter=lambda atoms: ~atoms.hetero
        )
        return _run_for_each_chain_pair(reference, pose, lrmsd_on_interfaces_only)

    def smaller_is_better(self) -> bool:
        return True


class InterfaceRMSD(Metric):
    """
    Compute the *Interface RMSD* for the given polymer complex as defined in [1]_.

    References
    ----------
    .. [1] https://doi.org/10.1093/bioinformatics/btae586
    """

    @property
    def name(self) -> str:
        return "iRMSD"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        reference, pose = filter_matched(
            reference, pose, prefilter=lambda atoms: ~atoms.hetero
        )
        # iRMSD is independent of the selection of receptor and ligand chain
        return _run_for_each_chain_pair(reference, pose, irmsd)

    def smaller_is_better(self) -> bool:
        return True


class ContactFraction(Metric):
    """
    Compute the fraction of correctly predicted reference contacts (*Fnat*) as defined
    in [1]_.

    References
    ----------
    .. [1] https://doi.org/10.1093/bioinformatics/btae586
    """

    @property
    def name(self) -> str:
        return "fnat"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        reference, pose = filter_matched(
            reference, pose, prefilter=lambda atoms: ~atoms.hetero
        )
        # Fnat is independent of the selection of receptor and ligand chain
        # Caution: `fnat()` returns both fnat and fnonnat -> select first element
        return _run_for_each_chain_pair(reference, pose, lambda *args: fnat(*args)[0])

    def smaller_is_better(self) -> bool:
        return False


class PocketAlignedLigandRMSD(Metric):
    """
    Compute the *Pocket aligned ligand RMSD* for the given PLI complex as defined
    in [1]_.

    References
    ----------
    .. [1] https://doi.org/10.1093/bioinformatics/btae586
    """

    @property
    def name(self) -> str:
        return "PLI-LRMSD"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        def run_lrmd(reference_chain1, reference_chain2, pose_chain1, pose_chain2):  # type: ignore[no-untyped-def]
            n_small_molecules = sum(
                is_small_molecule(chain)
                for chain in [reference_chain1, reference_chain2]
            )
            if n_small_molecules != 1:
                # Either two polymers or two small molecules -> not a valid PLI pair
                return np.nan
            return pocket_aligned_lrmsd(
                *_select_receptor_and_ligand(
                    reference_chain1, reference_chain2, pose_chain1, pose_chain2
                )
            )

        reference, pose = filter_matched(reference, pose)
        return _run_for_each_chain_pair(reference, pose, run_lrmd)

    def smaller_is_better(self) -> bool:
        return True


class BiSyRMSD(Metric):
    """
    Compute the *Binding-Site Superposed, Symmetry-Corrected Pose RMSD* (BiSyRMSD) for
    the given PLI complex.

    The method and default parameters are described in [1]_.

    Parameters
    ----------
    threshold : float
        The RMSD threshold to use for the *good* predictions.
    inclusion_radius : float, optional
        All residues where at least one heavy atom is within this radius of a heavy
        ligand atom, are considered part of the binding site.
    outlier_distance : float, optional
        The binding sites of the reference and pose are superimposed iteratively.
        In each iteration, atoms with a distance of more than this value are considered
        outliers and are removed in the next iteration.
        To disable outlier removal, set this value to ``inf``.
    max_iterations : int, optional
        The maximum number of iterations for the superimposition.
    min_anchors : int, optional
        The minimum number of anchors to use for the superimposition.
        If less than this number of anchors are present, the superimposition is
        performed on all interface backbone atoms.

    References
    ----------
    .. [1] https://doi.org/10.1002/prot.26601
    """

    def __init__(
        self,
        threshold: float,
        inclusion_radius: float = 4.0,
        outlier_distance: float = 3.0,
        max_iterations: int = 5,
        min_anchors: int = 3,
    ) -> None:
        self._threshold = threshold
        self._inclusion_radius = inclusion_radius
        self._outlier_distance = outlier_distance
        self._max_iterations = max_iterations
        self._min_anchors = min_anchors
        super().__init__()

    @property
    def name(self) -> str:
        return "BiSyRMSD"

    @property
    def thresholds(self) -> OrderedDict[str, float]:
        return OrderedDict(
            [(f"<{self._threshold}", 0), (f">{self._threshold}", self._threshold)]
        )

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        reference, pose = filter_matched(reference, pose)
        return bisy_rmsd(
            reference,
            pose,
            inclusion_radius=self._inclusion_radius,
            outlier_distance=self._outlier_distance,
            max_iterations=self._max_iterations,
            min_anchors=self._min_anchors,
        )

    def smaller_is_better(self) -> bool:
        return True


class BondLengthViolations(Metric):
    """
    Check for unusual bond lengths in the structure by comparing against reference
    values.
    Returns the percentage of bonds that are outside acceptable ranges.

    Parameters
    ----------
    tolerance : float, optional
        The relative tolerance for acceptable deviation from ideal bond length bounds.

    Notes
    -----
    Violations across residues are not considered.
    """

    def __init__(self, tolerance: float = 0.1) -> None:
        self._tolerance = tolerance
        super().__init__()

    @property
    def name(self) -> str:
        return "Bond length violations"

    @property
    def thresholds(self) -> OrderedDict[str, float]:
        return OrderedDict(
            [
                ("excellent", 0),
                ("good", 0.01),
                ("acceptable", 0.05),
                ("poor", 0.1),
            ]
        )

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        """
        Calculate the percentage of bonds that are outside acceptable ranges.

        Parameters
        ----------
        reference : AtomArray
            Not used in this metric as we compare against ideal bond lengths.
        pose : AtomArray
            The structure to evaluate.

        Returns
        -------
        float
            Percentage of bonds outside acceptable ranges (0.0 to 1.0).
        """
        if pose.array_length() == 0:
            return np.nan

        try:
            bounds = get_distance_bounds(pose)
        except struc.BadStructureError:
            return np.nan

        bond_indices = np.sort(pose.bonds.as_array()[:, :2], axis=1)
        if len(bond_indices) == 0:
            return np.nan
        bond_lengths = struc.index_distance(pose, bond_indices)
        # The bounds matrix has the lower bounds in the lower triangle
        # and the upper bounds in the upper triangle
        lower_bounds = bounds[bond_indices[:, 1], bond_indices[:, 0]]
        upper_bounds = bounds[bond_indices[:, 0], bond_indices[:, 1]]
        invalid_mask = (bond_lengths < lower_bounds * (1 - self._tolerance)) | (
            bond_lengths > upper_bounds * (1 + self._tolerance)
        )

        return float(
            np.count_nonzero(invalid_mask) / np.count_nonzero(np.isfinite(lower_bounds))
        )

    def smaller_is_better(self) -> bool:
        return True


class BondAngleViolations(Metric):
    """
    Check for unusual bond angles in the structure by comparing against
    idealized bond geometry.
    Returns the percentage of bonds that are outside acceptable ranges.

    This approach does not measure directly bond angles, but rather checks the
    distance between the atoms ``A`` and ``C`` in the bond angle ``ABC``.

    Parameters
    ----------
    tolerance : float, optional
        The relative tolerance for acceptable deviation from ideal distances.

    Notes
    -----
    Violations across residues are not considered.
    """

    def __init__(self, tolerance: float = 0.2) -> None:
        self._tolerance = tolerance
        super().__init__()

    @property
    def name(self) -> str:
        return "Bond angle violations"

    @property
    def thresholds(self) -> OrderedDict[str, float]:
        return OrderedDict(
            [
                ("excellent", 0),
                ("good", 0.01),
                ("acceptable", 0.05),
                ("poor", 0.1),
            ]
        )

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        """
        Calculate the percentage of bonds that are outside acceptable ranges.

        Parameters
        ----------
        reference : AtomArray
            Not used in this metric as we compare against ideal bond angles.
        pose : AtomArray
            The structure to evaluate.

        Returns
        -------
        float
            Percentage of bonds outside acceptable ranges (0.0 to 1.0).
        """
        if pose.array_length() == 0:
            return np.nan

        try:
            bounds = get_distance_bounds(pose)
        except struc.BadStructureError:
            return np.nan

        all_bonds, _ = reference.bonds.get_all_bonds()
        # For a bond angle 'ABC', this lost contains the atom indices for 'A' and 'C'
        bond_indices = []  # type: ignore[var-annotated]
        for bonded_indices in all_bonds:
            # Remove padding values
            bonded_indices = bonded_indices[bonded_indices != -1]
            bond_indices.extend(itertools.combinations(bonded_indices, 2))
        if len(bond_indices) == 0:
            return np.nan
        bond_indices = np.sort(bond_indices, axis=1)

        bond_lengths = struc.index_distance(pose, bond_indices)
        # The bounds matrix has the lower bounds in the lower triangle
        # and the upper bounds in the upper triangle
        lower_bounds = bounds[bond_indices[:, 1], bond_indices[:, 0]]
        upper_bounds = bounds[bond_indices[:, 0], bond_indices[:, 1]]
        invalid_mask = (bond_lengths < lower_bounds * (1 - self._tolerance)) | (
            bond_lengths > upper_bounds * (1 + self._tolerance)
        )

        return float(
            np.count_nonzero(invalid_mask) / np.count_nonzero(np.isfinite(lower_bounds))
        )

    def smaller_is_better(self) -> bool:
        return True


class RotamerViolations(Metric):
    """
    Check for the fraction of improbable amino acid rotamer angles,
    based on known crystal structures in the *Top8000* dataset [1]_.

    References
    ----------
    .. [1] https://doi.org/10.1002/prot.25039
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "Rotamer violations"

    @property
    def thresholds(self) -> OrderedDict[str, float]:
        # http://molprobity.biochem.duke.edu/help/validation_options/summary_table_guide.html
        return OrderedDict(
            [
                ("good", 0.000),
                ("warning", 0.003),
                ("bad", 0.015),
            ]
        )

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        """
        Calculate the percentage of residues with chi angles that are outside acceptable ranges.

        Parameters
        ----------
        reference : AtomArray
            Not used in this metric as we compare against ideal bond angles.
        pose : AtomArray
            The structure to evaluate.

        Returns
        -------
        float
            Percentage of bonds outside acceptable ranges (0.0 to 1.0).
        """
        if pose.array_length() == 0:
            return np.nan

        return get_fraction_of_rotamer_outliers(pose)

    def smaller_is_better(self) -> bool:
        return True


class RamachandranViolations(Metric):
    r"""
    Check for the fraction of improbable :math:`\phi`/:math:`\psi` angles,
    based on known crystal structures in the *Top8000* dataset [1]_.

    References
    ----------
    .. [1] https://doi.org/10.1002/prot.25039
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "Ramachandran violations"

    @property
    def thresholds(self) -> OrderedDict[str, float]:
        # http://molprobity.biochem.duke.edu/help/validation_options/summary_table_guide.html
        return OrderedDict(
            [
                ("good", 0.0000),
                ("warning", 0.0005),
                ("bad", 0.0050),
            ]
        )

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        """
        Calculate the percentage of bonds that are outside acceptable ranges.

        Parameters
        ----------
        reference : AtomArray
            Not used in this metric as we compare against ideal bond angles.
        pose : AtomArray
            The structure to evaluate.

        Returns
        -------
        float
            Percentage of bonds outside acceptable ranges (0.0 to 1.0).
        """

        if pose.array_length() == 0:
            return np.nan

        return get_fraction_of_rama_outliers(pose)

    def smaller_is_better(self) -> bool:
        return False


class ClashCount(Metric):
    """
    Count the number of clashes between atoms in the pose.
    """

    @property
    def name(self) -> str:
        return "Number of clashes"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        if pose.array_length() == 0:
            return np.nan
        return len(find_clashes(pose))

    def smaller_is_better(self) -> bool:
        return True


class PLIFRecovery(Metric):
    r"""
    Calculates the Polymer-Ligand Interaction Fingerprint (PLIF) recovery rate.

    This metric quantifies how well a predicted polymer-ligand pose structure
    recapitulates the specific interactions observed in a reference (e.g., crystal)
    structure [1]_:

    .. math::

        \textrm{PLIF Recovery} = \frac{\sum_{ir} \min(C_{ir}, P_{ir})}{\sum_{ir} C_{ir}},

    where :math:`C_ir` is the count of interaction type :math:`i` with receptor residue
    :math:`r` in the reference and :math:`P_ir` is the count of interaction type
    :math:`i` with receptor residue :math:`r` in the pose.

    Parameters
    ----------
    ph : float, optional
        The pH value used for charge estimation if relevant to contact definition.
    binding_site_cutoff : float, optional
        A cutoff used if contact definition involves focusing on a binding site.
    include_interactions : iterable of PLIFRecovery.InteractionType, optional
        The types of interactions to include in the PLIF calculations.
        By default, all of them are included.

    References
    ----------
    .. [1] https://doi.org/10.1186/s13321-025-01011-6
    """

    class InteractionType(Enum):
        """
        Defines the different contact types that can be evaluated.

        - ``HBOND_DONOR_RECEPTOR``: Hydrogen bond, where the donor atom is in the receptor.
        - ``HBOND_DONOR_LIGAND``: Hydrogen bond, where the donor atom is in the ligand.
        - ``HALOGEN_BOND``: Halogen bond.
        - ``PI_STACKING``: Pi stacking interaction.
        - ``CATION_PI``: Cation-Pi interaction, where the cation is in the receptor.
        - ``PI_CATION``: Cation-Pi interaction, where the cation is in the ligand.
        - ``IONIC_BOND``: Salt bridge.
        """

        HBOND_DONOR_RECEPTOR = auto()
        HBOND_DONOR_LIGAND = auto()
        HALOGEN_BOND = auto()
        PI_STACKING = auto()
        CATION_PI = auto()
        PI_CATION = auto()
        IONIC_BOND = auto()

    def __init__(
        self,
        ph: float = 7.4,
        binding_site_cutoff: float = 4.0,
        include_interactions: list["PLIFRecovery.InteractionType"] | None = None,
    ) -> None:
        self._ph = ph
        self._binding_site_cutoff = binding_site_cutoff
        if include_interactions is None:
            self._include_interactions = list(PLIFRecovery.InteractionType)
        else:
            self._include_interactions = include_interactions
        super().__init__()

    @property
    def name(self) -> str:
        return "PLIF Recovery"

    def _add_interactions_to_plifs(
        self,
        plifs: Dict[int, Dict["PLIFRecovery.InteractionType", int]],
        interaction_type: "PLIFRecovery.InteractionType",
        interactions: list | NDArray[np.int_],
        receptor: struc.AtomArray,
        mode: str,
    ) -> None:
        """
        Adds interactions to the PLIFs dictionary based on the specified mode.

        Parameters
        ----------
        plifs : dict
            The PLIF dictionary to modify.
        interaction_type : PLIFRecovery.InteractionType
            The type of interaction to increment.
        interactions : list or ndarray
            The interaction data. Format depends on the mode.
        receptor : AtomArray
            The receptor structure for residue lookup.
        mode : str
            The processing mode, either "atomic" or "ring".
        """
        if len(interactions) == 0:
            return

        if mode == "atomic":
            # Assumes 'interactions' is an NDArray of shape (n, 2)
            # Increments for each atom's residue in the contact list.
            interactions = np.array(interactions)
            receptor_indices = interactions[:, 0]
            res_ids = receptor.res_id[receptor_indices]
            for res_id in res_ids:
                plifs[res_id][interaction_type] += 1

        elif mode == "ring":
            assert isinstance(interactions, list)

            # Assumes 'interactions' is a list of interaction tuples.
            # Increments only once per residue for each ring system.
            for interaction in interactions:
                receptor_ring_indices = interaction[0]
                res_id = receptor.res_id[receptor_ring_indices[0]]
                plifs[res_id][interaction_type] += 1

    def _get_plifs_per_residue(
        self,
        receptor: struc.AtomArray,
        ligand: struc.AtomArray,
    ) -> Dict[int, Counter["PLIFRecovery.InteractionType"]]:
        """
        Generates a Polymer-Ligand Interaction Fingerprint dictionary where counts
        are aggregated per residue for each interaction type.
        """
        if ligand.array_length() == 0 or receptor.array_length() == 0:
            return {}

        # The original residue numbering between the reference and pose may be different
        # However the residue ID is important for this metric, as it is used to
        # identify the interactions
        # -> Create consistent residue numbering by just counting up
        receptor.res_id = struc.create_continuous_res_ids(
            receptor, restart_each_chain=False
        )

        contact_measurement = ContactMeasurement(
            receptor=receptor,
            ligand=ligand,
            cutoff=self._binding_site_cutoff,
            ph=self._ph,
        )

        # Initialize dict with ALL residues and ALL interaction types set to 0
        unique_res_ids = set(receptor.res_id)
        plifs: Dict[int, Counter["PLIFRecovery.InteractionType"]] = {
            res_id: Counter() for res_id in unique_res_ids
        }

        # --- Find All Interaction Types ---
        hbonds_rec_donor = contact_measurement.find_contacts_by_pattern(
            receptor_pattern=DONOR_PATTERN,
            ligand_pattern=ACCEPTOR_PATTERN,
            distance_scaling=HBOND_DISTANCE_SCALING,
        )
        hbonds_lig_donor = contact_measurement.find_contacts_by_pattern(
            receptor_pattern=ACCEPTOR_PATTERN,
            ligand_pattern=DONOR_PATTERN,
            distance_scaling=HBOND_DISTANCE_SCALING,
        )
        halogen_bonds = contact_measurement.find_contacts_by_pattern(
            receptor_pattern=ACCEPTOR_PATTERN,
            ligand_pattern=HALOGEN_PATTERN,
            distance_scaling=HALOGEN_DISTANCE_SCALING,
        )
        ionic_bonds = contact_measurement.find_salt_bridges()
        pi_stacking_interactions = contact_measurement.find_stacking_interactions()
        pi_cation_interactions = contact_measurement.find_pi_cation_interactions()

        # --- Populate PLIFs ---
        self._add_interactions_to_plifs(
            plifs,
            PLIFRecovery.InteractionType.HBOND_DONOR_RECEPTOR,
            hbonds_rec_donor,
            receptor,
            mode="atomic",
        )
        self._add_interactions_to_plifs(
            plifs,
            PLIFRecovery.InteractionType.HBOND_DONOR_LIGAND,
            hbonds_lig_donor,
            receptor,
            mode="atomic",
        )
        self._add_interactions_to_plifs(
            plifs,
            PLIFRecovery.InteractionType.HALOGEN_BOND,
            halogen_bonds,
            receptor,
            mode="atomic",
        )
        self._add_interactions_to_plifs(
            plifs,
            PLIFRecovery.InteractionType.IONIC_BOND,
            ionic_bonds,
            receptor,
            mode="atomic",
        )

        self._add_interactions_to_plifs(
            plifs,
            PLIFRecovery.InteractionType.PI_STACKING,
            pi_stacking_interactions,
            receptor,
            mode="ring",
        )

        # Handle the two types of pi-cation interactions separately
        for interaction in pi_cation_interactions:
            receptor_part, _, cation_in_receptor = interaction
            if cation_in_receptor:
                # Receptor has the cation (single atom), so use "atomic" mode
                atomic_contact = np.array([[receptor_part[0], -1]], dtype=int)
                self._add_interactions_to_plifs(
                    plifs,
                    PLIFRecovery.InteractionType.CATION_PI,
                    atomic_contact,
                    receptor,
                    mode="atomic",
                )
            else:
                # Receptor has the ring, so use "ring" mode
                self._add_interactions_to_plifs(
                    plifs,
                    PLIFRecovery.InteractionType.PI_CATION,
                    [interaction],
                    receptor,
                    mode="ring",
                )

        return plifs

    def _calculate_recovery_score(
        self,
        reference_plifs: Dict[int, Counter["PLIFRecovery.InteractionType"]],
        pose_plifs: Dict[int, Counter["PLIFRecovery.InteractionType"]],
    ) -> float:
        """
        Calculate PLIF recovery score using Errington et al. formula.
        PLIF Recovery = (sum_ir min(C_ir, P_ir)) / (sum_ir C_ir)
        """
        numerator = 0
        denominator = 0

        # Only iterate over residues and interactions present in the reference
        all_residues = reference_plifs.keys()
        all_interaction_types = self._include_interactions

        for res_id in all_residues:
            for interaction_type in all_interaction_types:
                c_ir = reference_plifs[res_id][interaction_type]
                p_ir = pose_plifs.get(res_id, Counter())[interaction_type]
                numerator += min(c_ir, p_ir)
                denominator += c_ir

        if denominator == 0:
            # If the reference has no contacts, the metric is undefined.
            return np.nan

        return numerator / denominator

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        if reference.array_length() == 0 or pose.array_length() == 0:
            return np.nan

        # Only the receptor residues need to be matched...
        ref_receptor, pose_receptor = filter_matched(
            reference, pose, prefilter=lambda atoms: ~atoms.hetero
        )
        # ... the ligands may be different in this metric
        ref_ligand = reference[reference.hetero]
        pose_ligand = pose[pose.hetero]

        # Only evaluate on PLI systems - check for both ligands and polymers
        if (
            ref_receptor.array_length() == 0
            or ref_ligand.array_length() == 0
            or pose_ligand.array_length() == 0
        ):
            return np.nan

        try:
            reference_plifs = self._get_plifs_per_residue(ref_receptor, ref_ligand)
            pose_plifs = self._get_plifs_per_residue(pose_receptor, pose_ligand)
        except struc.BadStructureError:
            return np.nan
        return self._calculate_recovery_score(reference_plifs, pose_plifs)

    def smaller_is_better(self) -> bool:
        return False

    @property
    def thresholds(self) -> OrderedDict[str, float]:
        return OrderedDict([("Low", 0.0), ("Medium", 0.5), ("High", 0.9)])


class ChiralityViolations(Metric):
    """
    Check for differences in the chirality of the reference and pose.
    """

    @property
    def name(self) -> str:
        return "Chirality-violation"

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        """
        Returns the fraction of chiral centers that have a different chirality
        in the reference as compared to the pose.

        Parameters
        ----------
        reference : AtomArray
            The reference structure of the system.
        pose : AtomArray
            The predicted pose.
            Must have the same length and atom order as the `reference`.

        Returns
        -------
        float
            The fraction of chiral centers that have a different chirality in the reference as compared to the pose.
        """
        reference, pose = filter_matched(reference, pose)
        if reference.array_length() == 0:
            return np.nan

        # Convert the reference and pose to RDKit molecules
        ref_mol = rdkit_interface.to_mol(reference, explicit_hydrogen=False)
        pose_mol = rdkit_interface.to_mol(pose, explicit_hydrogen=False)

        # Assign chiral centers
        Chem.AssignStereochemistryFrom3D(ref_mol)
        Chem.AssignStereochemistryFrom3D(pose_mol)

        # Get the chirality of the reference and pose
        ref_chirality = np.array([atom.GetChiralTag() for atom in ref_mol.GetAtoms()])
        pose_chirality = np.array([atom.GetChiralTag() for atom in pose_mol.GetAtoms()])

        chiral_count = np.count_nonzero(
            ref_chirality != int(Chem.ChiralType.CHI_UNSPECIFIED)
        )
        violation_count = np.count_nonzero(ref_chirality != pose_chirality)

        if chiral_count == 0:
            return np.nan

        return float(violation_count / chiral_count)

    def smaller_is_better(self) -> bool:
        return True


class PocketDistance(Metric):
    r"""
    Calculates the distance between the centroid of the reference ligand
    (i.e. the pocket center) and the pose in the ligand. [1]_

    If multiple pockets are present, the average distance is calculated.

    Parameters
    ----------
    use_pose_centroids : bool, optional
        If ``True``, the metric quantifies the distance between the pocket center and
        the pose ligand centroid (also called DCC [2]_).
        Otherwise, the metric is more permissive and takes the minimum distance of any
        pose ligand atom to the pocket center (also called DCA).

    Notes
    -----
    Note that for ``use_pose_centroids=False`` even a perfect match might not have
    a distance of zero, as centroid may not lie directly on some ligand atom directly.

    References
    ----------
    .. [1] https://doi.org/10.1073/pnas.0707684105
    .. [2] https://doi.org/10.1016/j.str.2011.02.015
    """

    def __init__(
        self,
        use_pose_centroids: bool = True,
    ) -> None:
        self._use_pose_centroids = use_pose_centroids
        super().__init__()

    @property
    def name(self) -> str:
        return "DCC" if self._use_pose_centroids else "DCA"

    @property
    def thresholds(self) -> OrderedDict[str, float]:
        # Based on the 'average radius of gyration for ligand molecules'
        # (https://doi.org/10.1073/pnas.0707684105)
        return OrderedDict([("good", 0.0), ("bad", 4.0)])

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        ref_receptor, pose_receptor = filter_matched(
            reference, pose, prefilter=lambda atoms: ~atoms.hetero
        )
        ref_ligand = reference[reference.hetero]
        pose_ligand = pose[pose.hetero]

        if (
            ref_receptor.array_length() == 0
            or ref_ligand.array_length() == 0
            or pose_ligand.array_length() == 0
        ):
            return np.nan

        # Superimpose on polymer chains
        _, transform = struc.superimpose(ref_receptor, pose_receptor)
        pose_ligand = transform.apply(pose_ligand)

        ref_ligands = list(struc.chain_iter(ref_ligand))
        pose_ligands = list(struc.chain_iter(pose_ligand))
        if len(ref_ligands) != len(pose_ligands):
            raise IndexError(
                f"Reference has {len(ref_ligands)} ligands, "
                f"but pose has {len(pose_ligands)} ligands"
            )
        ref_centroids = np.array([struc.centroid(lig) for lig in ref_ligands])
        pose_centroids = np.array([struc.centroid(lig) for lig in pose_ligands])
        ref_ligand_order, pose_ligand_order = find_matching_centroids(
            ref_centroids, pose_centroids
        )

        if self._use_pose_centroids:
            ref_centroids = ref_centroids[ref_ligand_order]
            pose_centroids = pose_centroids[pose_ligand_order]
            return np.mean(
                np.linalg.norm(pose_centroids - ref_centroids, axis=1)
            ).item()
        else:
            ref_ligands = [ref_ligands[i] for i in ref_ligand_order]
            pose_ligands = [pose_ligands[i] for i in pose_ligand_order]
            min_distances = np.array(
                [
                    np.min(struc.distance(pose_ligand, pocket_center))
                    for pose_ligand, pocket_center in zip(pose_ligands, ref_centroids)
                ]
            )
            return np.mean(min_distances).item()

    def smaller_is_better(self) -> bool:
        return True


class PocketVolumeOverlap(Metric):
    r"""
    Calculates the *discretized volume overlap* (DVO) between the reference and pose
    ligand. [1]_

    It is defined as the intersection of the reference and pose ligand volume
    divided by the union of the volumes.
    The volume of an atom is a sphere with radius equal to the *Van-der-Waals* radius.
    If multiple ligands are present, the average DVO is calculated.

    Parameters
    ----------
    voxel_size : float, optional
        The size of the voxels used for the DVO calculation.
        The computation becomes more accurate with smaller voxel sizes, but
        the run time scales inverse cubically with voxel size.

    References
    ----------
    .. [1] https://doi.org/10.1016/j.str.2011.02.015
    """

    def __init__(
        self,
        voxel_size: float = 0.5,
    ) -> None:
        self._voxel_size = voxel_size
        super().__init__()

    @property
    def name(self) -> str:
        return "DVO"

    @property
    def thresholds(self) -> OrderedDict[str, float]:
        # Based on the 'average radius of gyration for ligand molecules'
        # (https://doi.org/10.1073/pnas.0707684105)
        return OrderedDict([("low", 0.0), ("high", 0.5)])

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        ref_receptor, pose_receptor = filter_matched(
            reference, pose, prefilter=lambda atoms: ~atoms.hetero
        )
        ref_ligand = reference[reference.hetero]
        pose_ligand = pose[pose.hetero]

        if (
            ref_receptor.array_length() == 0
            or ref_ligand.array_length() == 0
            or pose_ligand.array_length() == 0
        ):
            return np.nan

        # Superimpose on polymer chains
        _, transform = struc.superimpose(ref_receptor, pose_receptor)
        pose_ligand = transform.apply(pose_ligand)

        ref_ligands = list(struc.chain_iter(ref_ligand))
        pose_ligands = list(struc.chain_iter(pose_ligand))
        if len(ref_ligands) != len(pose_ligands):
            raise IndexError(
                f"Reference has {len(ref_ligands)} ligands, "
                f"but pose has {len(pose_ligands)} ligands"
            )
        ref_centroids = np.array([struc.centroid(lig) for lig in ref_ligands])
        pose_centroids = np.array([struc.centroid(lig) for lig in pose_ligands])
        ref_ligand_order, pose_ligand_order = find_matching_centroids(
            ref_centroids, pose_centroids
        )
        ref_ligands = [ref_ligands[i] for i in ref_ligand_order]
        ref_centroids = [ref_centroids[i] for i in ref_ligand_order]
        pose_ligands = [pose_ligands[i] for i in pose_ligand_order]
        pose_centroids = [pose_centroids[i] for i in pose_ligand_order]

        dvo = []
        for ref_ligand, ref_centroid, pose_ligand, pose_centroid in zip(
            ref_ligands, ref_centroids, pose_ligands, pose_centroids
        ):
            # Shortcut: If the two molecules are two far away of each other, we do not
            # need to perform costly volume calculations using a large voxel grid
            ref_radius = np.max(struc.distance(ref_ligand, ref_centroid))
            pose_radius = np.max(struc.distance(pose_ligand, pose_centroid))
            if ref_radius + pose_radius < struc.distance(ref_centroid, pose_centroid):
                dvo.append(0.0)
            else:
                _, intersection_volume, union_volume = volume_overlap(
                    [ref_ligand, pose_ligand], self._voxel_size
                )
                dvo.append(intersection_volume / union_volume)

        return np.mean(dvo).item()

    def smaller_is_better(self) -> bool:
        return False


class LigandValenceViolations(Metric):
    r"""
    Counts the total number atoms with valance violations over all ligands using
    RDKit's internal valence checks.

    This metric finds valence violations that cannot be fixed by kekulization or
    adding or removing charges. It is useful for identifying ligands that are
    likely to be problematic.
    """

    @property
    def name(self) -> str:
        return "Ligand valence violations"

    def smaller_is_better(self) -> bool:
        return True

    def evaluate(self, reference: struc.AtomArray, pose: struc.AtomArray) -> float:
        isolated_ligand_masks = _select_isolated_ligands(pose)
        ligand_atomarrays = [pose[mask] for mask in isolated_ligand_masks]
        num_violations_per_ligand = [
            _count_valence_violations(aarray) for aarray in ligand_atomarrays
        ]
        return np.sum(num_violations_per_ligand).item()


def _count_valence_violations(ligand: struc.AtomArray) -> int:
    mol = rdkit_interface.to_mol(ligand, explicit_hydrogen=False)
    try:
        sanitize(mol)
    except Exception:
        return sum(atom.HasValenceViolation() for atom in mol.GetAtoms())

    return 0


def _run_for_each_monomer(
    reference: struc.AtomArray,
    pose: struc.AtomArray,
    function: Callable[[struc.AtomArray, struc.AtomArray], float],
) -> float:
    """
    Run the given function for each monomer in the reference and pose.

    Parameters
    ----------
    reference : AtomArray, shape=(n,)
        The reference structure of the system.
    pose : AtomArray, shape=(n,)
        The predicted pose.
        Must have the same length and atom order as the `reference`.
    function : Callable
        The function to run for each monomer.
        Takes the reference and pose as input and returns a scalar value.

    Returns
    -------
    metrics : float
        The average return value of `function`, weighted by the number of atoms.
        If the input structure contains no chains, *NaN* is returned.
    """
    values = []
    chain_starts = struc.get_chain_starts(reference, add_exclusive_stop=True)
    for start_i, stop_i in itertools.pairwise(chain_starts):
        reference_chain = reference[start_i:stop_i]
        pose_chain = pose[start_i:stop_i]
        values.append(function(reference_chain, pose_chain))
    values = np.array(values)

    # Ignore chains where the values are NaN
    not_nan_mask = np.isfinite(values)
    values = values[not_nan_mask]
    if len(values) == 0:
        # No chains in the structure
        return np.nan
    else:
        n_atoms_per_chain = np.diff(chain_starts)
        return np.average(values, weights=n_atoms_per_chain[not_nan_mask]).item()


def _run_for_each_chain_pair(
    reference: struc.AtomArray,
    pose: struc.AtomArray,
    function: Callable[
        [struc.AtomArray, struc.AtomArray, struc.AtomArray, struc.AtomArray], float
    ],
) -> float:
    """
    Run the given function for each chain pair combination in the reference and pose
    and return the average value.

    Parameters
    ----------
    reference : AtomArray, shape=(n,)
        The reference structure of the system.
    pose : AtomArray, shape=(n,)
        The predicted pose.
        Must have the same length and atom order as the `reference`.
    function : Callable[(reference_chain1, reference_chain2, pose_chain1, pose_chain2), float]
        The function to run for each interface.
        Takes the reference and pose chains in contact as input and returns a scalar
        value.
        The function may also return *NaN*, if its result should be ignored.

    Returns
    -------
    metrics : float
        The average return value of `function`, weighted by the number of atoms.
        If the input structure contains only one chain, *NaN* is returned.
    """
    chain_starts = struc.get_chain_starts(reference, add_exclusive_stop=True)
    reference_chains = [
        reference[start:stop] for start, stop in itertools.pairwise(chain_starts)
    ]
    pose_chains = [pose[start:stop] for start, stop in itertools.pairwise(chain_starts)]
    results = []
    for chain_i, chain_j in itertools.combinations(range(len(reference_chains)), 2):
        results.append(
            function(
                reference_chains[chain_i],
                reference_chains[chain_j],
                pose_chains[chain_i],
                pose_chains[chain_j],
            )
        )
    if np.isnan(results).all():
        return np.nan
    return np.nanmean(results).item()


def _average_over_ligands(
    reference: struc.AtomArray,
    pose: struc.AtomArray,
    function: Callable[[struc.AtomArray, struc.AtomArray], float],
) -> float:
    """
    Run the given function for each ligand in the reference and pose.
    and return the average value.

    Parameters
    ----------
    reference : AtomArray, shape=(n,)
        The reference structure of the system.
    pose : AtomArray, shape=(n,)
        The predicted pose.
        Must have the same length and atom order as the `reference`.
    function : Callable[[reference_ligand, pose_ligand], float]
        The function to run for each ligand.
        Takes the reference and pose ligand as input and returns a scalar
        value.

    Returns
    -------
    metrics : float
        The average return value of `function`, weighted by the number of atoms.
        If the input structure contains no ligand atoms, *NaN* is returned.
    """
    values = _run_for_each_ligand(reference, pose, function)
    if np.isnan(values).all():
        return np.nan
    else:
        return np.nanmean(values).item()


def _run_for_each_ligand(
    reference: struc.AtomArray,
    pose: struc.AtomArray,
    function: Callable[[struc.AtomArray, struc.AtomArray], Any],
) -> list[Any]:
    """
    Run the given function for each isolated ligand in complex with all polymers from
    the system.

    Parameters
    ----------
    reference : AtomArray, shape=(n,)
        The reference structure of the system.
    pose : AtomArray, shape=(n,)
        The predicted pose.
        Must have the same length and atom order as the `reference`.
    function : Callable[[reference_ligand, pose_ligand], float]
        The function to run.
        Takes the reference and pose system as input and returns a scalar
        value.

    Returns
    -------
    metrics : float
        The average return value of `function`.
        If the input structure contains no ligand atoms, *NaN* is returned.
    """
    values = []
    ligand_mask = reference.hetero
    polymer_mask = ~ligand_mask
    chain_starts = struc.get_chain_starts(reference)
    if len(chain_starts) == 0:
        # No chains in the structure
        return []
    chain_masks = struc.get_chain_masks(reference, chain_starts)
    # Only keep chain masks that correspond to ligand chains
    ligand_masks = chain_masks[(chain_masks & ligand_mask).any(axis=-1)]
    for ligand_mask in ligand_masks:
        # Evaluate each isolated ligand in complex separately
        complex_mask = ligand_mask | polymer_mask
        values.append(function(reference[complex_mask], pose[complex_mask]))
    return values


def _select_receptor_and_ligand(
    reference_chain1: struc.AtomArray,
    reference_chain2: struc.AtomArray,
    pose_chain1: struc.AtomArray,
    pose_chain2: struc.AtomArray,
) -> tuple[struc.AtomArray, struc.AtomArray, struc.AtomArray, struc.AtomArray]:
    """
    Select the receptor and ligand for the given interface.

    The longer of both chains is the receptor.

    Parameters
    ----------
    reference_chain1, reference_chain2 : AtomArray, shape=(n,)
        The reference chains.
    pose_chain1, pose_chain2 : AtomArray, shape=(n,)
        The pose chains.

    Returns
    -------
    reference_receptor, reference_ligand, pose_receptor, pose_ligand : AtomArray
        The selected receptor and ligand.
    """
    if is_small_molecule(reference_chain1):
        return (reference_chain2, reference_chain1, pose_chain2, pose_chain1)
    elif is_small_molecule(reference_chain2):
        return (reference_chain1, reference_chain2, pose_chain1, pose_chain2)
    elif reference_chain1.array_length() >= reference_chain2.array_length():
        return (reference_chain1, reference_chain2, pose_chain1, pose_chain2)
    else:
        return (reference_chain2, reference_chain1, pose_chain2, pose_chain1)


def _select_isolated_ligands(pose: struc.AtomArray) -> NDArray[np.bool_]:
    """
    Returns masks that select ligannds in the system without polymer chains.

    Parameters
    ----------
    pose : AtomArray, shape=(n,)
        The system to select from.

    Returns
    -------
    NDArray[bool], shape=(n_subsets, n)
        The masks that can be used to select isolated ligands in the system.
        The first dimension of the array is the subset index.
    """
    ligand_mask = pose.hetero
    chain_starts = struc.get_chain_starts(pose)
    if len(chain_starts) == 0:
        # No chains in the structure
        return np.array([], dtype=bool)
    chain_masks = struc.get_chain_masks(pose, chain_starts)
    # Only keep chain masks that correspond to ligand chains
    ligand_masks = chain_masks[(chain_masks & ligand_mask).any(axis=-1)]
    return ligand_masks
